import { createContext, useContext, useReducer, type ReactNode, type Dispatch } from 'react'
import { reducer, type Action } from './reducer.ts'
import { initialState, type AppState } from '../types/state.ts'

const StateContext = createContext<AppState>(initialState)
const DispatchContext = createContext<Dispatch<Action>>(() => {})

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState)
  return (
    <StateContext.Provider value={state}>
      <DispatchContext.Provider value={dispatch}>
        {children}
      </DispatchContext.Provider>
    </StateContext.Provider>
  )
}

export function useAppState() {
  return useContext(StateContext)
}

export function useAppDispatch() {
  return useContext(DispatchContext)
}
