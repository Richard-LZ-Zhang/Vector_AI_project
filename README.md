# Vector_AI_project

Problem:
The CNN_Server gcloud: now once receive one image, then compute and publish it, which is ineffficient. Improvement: Collect a batch of result and publish.
The CNN Server and Sender gcloud: the self.service.futures holds all the future (which means one future per message and stays in the list permanently), which might be problematic