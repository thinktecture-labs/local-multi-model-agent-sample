# On-Device SLM Tool Calling — Lessons Learned

Insights from building and stress-testing the LocalLife on-device AI agent using
**LFM 2.5 1.2B Instruct (Q4_K_M GGUF)** via the **LEAP SDK** on iPhone (A17 Pro).

---

## 1. The Core Challenge

A 1.2B parameter model must reliably:
1. **Select exactly one tool** from three registered functions per user query
2. **Pass correct arguments** to the selected tool
3. **Synthesize a natural-language response** from the tool's JSON results

With naive setup, tool selection was correct only **35-80% of the time** depending on the query.
After the fixes documented here, tool selection is **100% consistent across 20/20 stress test runs**.

---

## 2. Critical Insight: Conversation History Breaks Tool Selection

### The Problem

The LEAP SDK's `Conversation` object accumulates chat history across
`generateResponse()` calls. The `GenerationOptions.resetHistory` flag (defaults to `true`)
resets the KV cache before each generation, but the conversation's internal `history` array
still grows.

**With a 1.2B model, accumulated history causes tool selection to degrade:**

- The model sees prior tool results and "remembers" answers, sometimes skipping tools entirely
  ("Based on your recent calendar search...")
- Prior tool results containing health keywords leak into calendar queries, confusing routing
- The model occasionally refuses to call tools ("I don't have access to your calendar data")

### The Fix: Fresh Conversation Per Query

```swift
// AgentLoop.swift — at the start of each run()
modelManager.resetConversation()
```

```swift
// ModelManager.swift
public func resetConversation() {
    guard let runner = modelRunner else { return }
    let conv = runner.createConversation(systemPrompt: AgentConfiguration.systemPrompt)
    for fn in registeredFunctions {
        conv.registerFunction(fn)
    }
    self.conversation = conv
}
```

**Why `resetHistory = true` alone is insufficient:** Even with KV cache reset, the 1.2B model's
attention is influenced by the conversation history tokens present in the context. Creating a
fresh `Conversation` object guarantees zero history bleed between queries.

**Chat history lives at the app layer** (`ChatViewModel.messages`), not in the SDK's
`Conversation` object. The user sees their full conversation in the UI; the model just doesn't
carry internal state between queries. With a 1.2B model, this is the right tradeoff.

### Test Evidence

| Query                                     | Without Reset | With Reset |
|-------------------------------------------|:------------:|:----------:|
| "What appointments do I have this week?"  | 2/5          | **5/5**    |
| "Show my heart rate trends for 30 days"   | 3/5          | **5/5**    |
| "Search my reminders for health questions"| 4/5          | **5/5**    |
| "Prepare me for my meeting with Dr. ..."  | 5/5          | **5/5**    |

---

## 3. Two-Round Agent Loop with `resetHistory` Control

The agent loop uses two inference rounds per query:

```
Round 1 (Tool Selection):  resetHistory = true  (default — fresh KV cache)
  User query → Model generates → tool call emitted

  [Tool executes, results collected]

Round 2 (Synthesis):       resetHistory = false  (preserves Round 1 context)
  Tool results → Model generates → natural language answer
```

### Why Round 2 needs `resetHistory = false`

Without preserving Round 1 context, the synthesis round receives tool results but has
**no knowledge of what the user originally asked**. The model sees raw JSON data with no
question context, producing poor or empty responses.

```swift
// AgentLoop.swift
var synthesisOptions = options
synthesisOptions.resetHistory = false  // Keep user question + tool call in context
let synthStream = modelManager.generateResponse(message: toolResultsMessage, options: synthesisOptions)
```

---

## 4. System Prompt Design for Small Models

### What Works

```
You are LocalLife, an AI assistant with three tools.
You MUST call exactly one tool for every question.

TOOL ROUTING:
- search_calendar → calendar, appointment, schedule, event, meeting, this week
- query_health_data → health, heart rate, steps, blood pressure, weight
- search_reminders → reminder, to-do, todo, task, checklist

IMPORTANT: "calendar" or "event" always means search_calendar, never search_reminders.
After the tool returns results, summarize with bullet points and include units for health data.
```

### Key Principles for 1.2B Models

1. **Keep it short.** Every token in the system prompt competes for the model's limited
   attention budget. Verbose prompts cause the model to "forget" rules mid-generation.

2. **Keyword → tool mapping, not descriptions.** The model matches keywords better than
   it interprets semantic descriptions. List the exact words that map to each tool.

3. **Add negative disambiguation.** Explicitly state what a keyword does NOT mean:
   `"calendar" or "event" always means search_calendar, never search_reminders.`
   Without this, "calendar events" was routed to `search_reminders` 40% of the time.

4. **Don't duplicate tool definitions.** The LEAP SDK injects tool schemas via
   `registerFunction()`. Repeating tool descriptions in the system prompt gives the model
   two conflicting representations and degrades tool selection accuracy.

5. **"MUST call a tool" in the opening sentence.** The model pays more attention to the
   beginning of the system prompt. Burying the instruction in a numbered list reduces
   compliance.

---

## 5. Generation Parameters

```swift
temperature: 0.0       // Fully deterministic — same input → same tool selection
repetitionPenalty: 1.05 // Mild penalty to prevent degenerate repetition
maxTokens: 600         // Budget for synthesis responses
```

- **Temperature 0.0** is essential for consistent tool selection. Any temperature > 0
  introduces randomness that causes the model to occasionally pick the wrong tool.
- **Repetition penalty 1.05** prevents the model from generating repetitive text in
  synthesis without being aggressive enough to distort tool call formatting.

---

## 6. ReWOO Pattern for Multi-Tool Queries

The 1.2B model cannot reliably plan and execute multi-tool orchestration (e.g., "prepare me
for my meeting" requiring calendar + health + reminders). It either calls only one tool or
calls all three with incorrect arguments.

### Solution: Programmatic Preflight (ReWOO Pattern)

For detected multi-tool queries (meeting prep), skip the model's tool selection entirely:

1. **Detect** the intent in code (`isMeetingPrepQuery()` — keyword matching)
2. **Execute** all three tools programmatically with predefined arguments
3. **Synthesize** by giving the model all tool results in a single prompt

```swift
if isMeetingPrep {
    let toolResults = await executePreflightTools(onEvent: onEvent)
    currentMessage = """
    The user asked: \(message)
    Here are the results from their personal data:
    \(toolResults)
    Summarize this as a meeting preparation briefing.
    """
}
```

This achieves **5/5 consistency** because the model only does synthesis (its strength),
not multi-tool planning (its weakness).

---

## 7. Guard Against Multi-Tool Over-Firing

For single-tool queries, the model sometimes emits multiple tool calls. The `bestMatch()`
function selects the most relevant one by comparing query words against tool metadata:

```swift
if !isMeetingPrep && pendingToolCalls.count > 1 {
    pendingToolCalls = [bestMatch(pendingToolCalls, forQuery: message)]
}
```

The scoring uses word-level intersection between the user query and each tool's name,
description, and parameter descriptions from the registry — adapts automatically if tools
change.

---

## 8. LEAP SDK Key APIs

### `GenerationOptions`

| Property           | Type      | Default | Notes                                    |
|--------------------|-----------|---------|------------------------------------------|
| `temperature`      | `Float?`  | `nil`   | Set to 0.0 for deterministic output      |
| `repetitionPenalty` | `Float?` | `nil`   | 1.05 works well                          |
| `resetHistory`     | `Bool`    | `true`  | Clears KV cache before generation        |
| `maxOutputTokens`  | `UInt32?` | `nil`   | Limits generation length                 |
| `sequenceLength`   | `UInt32?` | `nil`   | Context window size                      |

### `Conversation`

- `history: [ChatMessage] { get }` — read-only, grows with each generation
- `registerFunction(_ function: LeapFunction)` — registers a tool
- `generateResponse(userTextMessage:generationOptions:)` — streaming generation
- No public `clearHistory()` method — must create a new conversation to reset

### `ModelRunner`

- `createConversation(systemPrompt:)` — creates a new conversation (cheap, no model reload)
- `createConversationFromHistory(history:)` — creates from existing messages
- `unload()` — releases model resources (triggers GGML crash — see section 9)

---

## 9. llama.cpp Metal Exit Crash (GGML_ASSERT)

### The Problem

When the process exits after running inference, llama.cpp's C++ global destructor
`ggml_metal_device_deleteGlobal` fires and asserts `[rsets->data count] == 0`. This
assertion fails because Metal result sets are still allocated, causing a `SIGABRT` that
traps the debugger and hangs the test runner.

**This is a llama.cpp bug, not fixable from Swift.**

### Attempted Fixes That Don't Work

| Approach                        | Why It Fails                                              |
|---------------------------------|-----------------------------------------------------------|
| `runner.unload()`               | Triggers the same GGML Metal cleanup → crash              |
| `Unmanaged.passRetained()`      | C++ global destructors run regardless of Swift retain count|
| `signal(SIGABRT) { _exit(0) }` | LLDB intercepts signals before our handler                |

### The Fix: `atexit { _exit(0) }`

Register an `atexit` handler during model setup. `atexit` handlers run in LIFO order
during normal process exit, and they execute **before** C++ static destructors. Since we
register ours after llama.cpp has initialized, ours fires first (LIFO), calling `_exit(0)`
which terminates immediately — skipping C++ destructors entirely.

```swift
// In test model setup, after model is loaded:
atexit { _exit(0) }
```

This is only needed in test targets. The app itself doesn't exit in a way that triggers
the destructors (iOS suspends apps, doesn't exit them).

---

## 10. On-Device Performance (A17 Pro, iPhone 15 Pro)

| Metric                | Value           |
|-----------------------|-----------------|
| Model file            | ~700 MB (Q4_K_M)|
| Model load time       | ~5 seconds      |
| Prompt eval speed     | ~600 tok/s      |
| Generation speed      | ~50 tok/s       |
| Single tool query     | ~5-10 seconds   |
| Meeting prep (3 tools)| ~15-20 seconds  |

---

## 11. Testing Strategy

### Test Isolation

Tests run inside the app's test host process. To prevent the app's `ChatViewModel` from
competing for GPU/Metal resources:

```swift
// ChatViewModel.swift
func setup() async {
    if ProcessInfo.processInfo.environment["XCTestBundlePath"] != nil {
        return  // Skip app model setup when running tests
    }
    // ... normal app setup
}
```

### Shared Model Across Tests

Model loading is expensive (~5s). Tests use a static actor to load once and share:

```swift
private actor TestModelHolder {
    static let shared = TestModelHolder()
    // ... lazy setup with ModelManager, ToolRegistry, AgentLoop
}
```

### Stress Tests

Each of the 4 demo queries runs 5 times consecutively. The test passes if >= 3/5 runs
select the correct tool (threshold allows for occasional model variance, though current
setup achieves 5/5 consistently).

The stress report is embedded in the `#expect` comment — it only surfaces in test output
when the test actually fails, keeping passing runs clean.

---

## 12. Summary of All Fixes Applied

| Fix | File | What Changed |
|-----|------|-------------|
| Fresh conversation per query | `AgentLoop.swift` | `modelManager.resetConversation()` at start of `run()` |
| Synthesis preserves context | `AgentLoop.swift` | `synthesisOptions.resetHistory = false` for Round 2 |
| Keyword-based system prompt | `AgentConfiguration.swift` | Explicit keyword→tool mapping with negative disambiguation |
| Registered functions tracking | `ModelManager.swift` | `registeredFunctions` array for re-registration on reset |
| GGML exit crash workaround | Test files | `atexit { _exit(0) }` during model setup |
| Test host isolation | `ChatViewModel.swift` | `XCTestBundlePath` guard to skip app model loading |
