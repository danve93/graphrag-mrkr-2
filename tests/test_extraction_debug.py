
import asyncio
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.getcwd())

from core.entity_extraction import EntityExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_extraction():
    print("Initializing EntityExtractor...")
    extractor = EntityExtractor()
    
    # Text from Chap8 (Chunk 9)
    text = """3 ---
AdministrationGuide,Release25.9.0
8.1.2 MTAComponent
This section provides details about the Carbonio MTA’s mail queues, how to interact with them, and how to extract
relevant information.
MailQueueManagement
There are cases in which messages that users send remain stuck in the outgoing queue and can not be delivered to the
recipient. When this happens, you can check the MTA queue in different ways:
• by executing some commands from CLI
• from the Carbonio Admin Panel (see SectionQueue)
• by using Prometheus/Grafana based dashboards, on which suitable alarms can be defined
CLIcommandsavailable
The following is a list of commands that can be used from the CLI. you can check their manpage for their use and
configuration.
• postconfis a Postfix command to view or modify the Postfix configuration
• postfixis the main Postfix command, used to start, stop, and reload the service; flush the queues; check and
upgrade the configuration
• qshapeallows to examine Postfix queues in relation to time and sender’s or recipient’s domains
• postqueueis used to manage queues
• postsuperallows to delete messages from queues (must be run as root)"""

    chunk = {
        "chunk_id": "test_chunk_1",
        "content": text,
        "chunk_index": 0,
    }
    
    print("Extracting entities...")
    try:
        entities, relationships = await extractor.extract_from_chunks([chunk])
        
        print(f"\nFound {len(entities)} entities:")
        for name, ent in entities.items():
            print(f" - {name} ({ent.type})")
            
        print(f"\nFound {len(relationships)} relationship groups.")
    except Exception as e:
        print(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_extraction())
