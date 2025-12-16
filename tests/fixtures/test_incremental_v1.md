# Test Document V1 - Incremental Update Testing

## Introduction

This is a test document for verifying the incremental document update feature.
This section contains information that will remain unchanged between versions.

## Section A - Static Content

This paragraph will stay the same in version 2 of the document.
It contains static content that should be identified by the content hash algorithm.
The chunking should preserve this section across updates.

## Section B - Modified Content

This is the original content of Section B.
In version 2, this section will be completely rewritten.
The system should detect this as a changed chunk.

## Section C - Another Static Section

More static content that won't change.
This helps verify that unchanged chunks are correctly preserved.
Additional text to ensure the chunk is substantial enough.

## Conclusion

Final section with static content.
This paragraph also remains unchanged between versions.
