#!/bin/bash
# Fix corrupted BUNDLE_LOADER entries in project.pbxproj
# Run this AFTER quitting Xcode, then reopen the project.

PBXPROJ="LocalLife.xcodeproj/project.pbxproj"

# Remove lines with corrupted BUNDLE_LOADER (containing zero-width spaces)
# and the BUNDLE_LOADER[arch=*] duplicates, and NEW_SETTING
# Then add clean BUNDLE_LOADER = "$(TEST_HOST)"; in the right places

python3 -c "
import re

with open('$PBXPROJ', 'rb') as f:
    data = f.read()

# Remove lines containing corrupted BUNDLE_LOADER (with zero-width spaces U+200B = e2 80 8b)
lines = data.split(b'\n')
clean_lines = []
for line in lines:
    # Skip corrupted BUNDLE_LOADER lines (contain e2808b zero-width space bytes)
    if b'BUNDLE_LOADER' in line and b'\xe2\x80\x8b' in line:
        continue
    # Skip BUNDLE_LOADER[arch=*] lines (the workaround entries)
    if b'BUNDLE_LOADER[arch' in line:
        continue
    # Skip NEW_SETTING lines
    if b'NEW_SETTING' in line:
        continue
    clean_lines.append(line)

# Now add clean BUNDLE_LOADER after CODE_SIGN_STYLE in the test target configs
result = []
for line in clean_lines:
    result.append(line)
    # Add BUNDLE_LOADER right after CODE_SIGN_STYLE in test target build settings
    # We detect this by checking if the previous context is the test target config
    if b'CODE_SIGN_STYLE = Automatic;' in line:
        # Check if next lines are for test target (has TEST_HOST nearby)
        idx = len(result) - 1
        # Look ahead in clean_lines to see if TEST_HOST is within ~15 lines
        remaining_idx = clean_lines.index(line)
        look_ahead = b'\n'.join(clean_lines[remaining_idx:remaining_idx+20])
        if b'TEST_HOST' in look_ahead and b'BUNDLE_LOADER' not in look_ahead:
            indent = b'\t\t\t\t'
            result.append(indent + b'BUNDLE_LOADER = \"\$(TEST_HOST)\";')

data_out = b'\n'.join(result)

with open('$PBXPROJ', 'wb') as f:
    f.write(data_out)

print('Fixed! Corrupted BUNDLE_LOADER entries removed, clean ones added.')
"

echo ""
echo "Done. Now reopen LocalLife.xcodeproj in Xcode."
