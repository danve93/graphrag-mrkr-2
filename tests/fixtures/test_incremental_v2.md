# Test Document V2 - Incremental Update Testing

## Introduction

This is a test document for verifying the incremental document update feature.
This section contains information that will remain unchanged between versions.

## Section A - Static Content

This paragraph will stay the same in version 2 of the document.
It contains static content that should be identified by the content hash algorithm.
The chunking should preserve this section across updates.

## Section B - Modified Content

THIS SECTION HAS BEEN COMPLETELY REWRITTEN IN VERSION 2.
The content here is entirely different from the original.
This represents a significant change that should trigger re-processing.
New entities and relationships should be extracted from this section.

## Section C - Another Static Section

More static content that won't change.
This helps verify that unchanged chunks are correctly preserved.
Additional text to ensure the chunk is substantial enough.

## Section D - New Section

This is a brand new section added in version 2.
It did not exist in the original document.
The system should process this as a new chunk.

## Conclusion

Final section with static content.
This paragraph also remains unchanged between versions.
