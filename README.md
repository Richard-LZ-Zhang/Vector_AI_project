# Vector_AI_project

Problem and Proposed Solutions:
For Gcloud, problematic if there a lot of unacknowledged data in the cloud and then a subscriber connects to it. Solution: a "max_messages" option for pull
The CNN_Server gcloud: now once receive one image, then compute and publish it, which is ineffficient. Improvement: Collect a batch of result and publish, as the result is only an integer under 10 (1 byte will suffice)
The CNN Server and Sender gcloud: the self.service.futures holds all the future (which means one future per message and stays in the list permanently), which might be problematic
auto reconnect