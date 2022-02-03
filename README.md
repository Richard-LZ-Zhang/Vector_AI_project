# Vector_AI_project
I. What I did
I built a CNN model (see ml/cnn.py) that trains on Fashion_MNIST dataset and reaches test accurary of 90.0%. This model and relevent dataset objects also supports numpy arra (dtype uint8) input (tested in test_cnn_np).
I also built the server api (see dataset_api module) for cnn server, receiver, and sender, and for both Kafka and Gcloud. The configuration is stored in servoce_config dir. Google cloud could be directly run as I have had the keys attached.

II. How to use?
Run main.py file and there will be three threads for sender receiver and cnn server running. You may change the image_num (num of images sent) and MODE ("gcloud" / "kafka") variable to see what happens. The three "servers" could also be run seprately in receiver_start.py, sender_start.py, servert_start.py

Run train_store completes a full training on Fasion_MNIST data, and store the model parameters in a path.

III Problem and Proposed Solutions:
1. No auto reconnect for gcloud and Kafka strategies
2. For Gcloud, problematic if there a lot of unacknowledged data in the cloud and then a subscriber connects to it. Solution: a "max_messages" option for pull
3. The CNN_Server both gcloud and kafka: now once receive one image, then compute and publish it, which is inefficient. Improvement: Collect a batch of result and publish, as the result is only an integer under 10 (1 byte will suffice)
4. The CNN Server and Sender gcloud: the self.service.futures holds all the future (which means one future per message and stays in the list permanently), which might be problematic
6. good to see a python venv
7. Documentation and Annotation are poor, due to lack of time. But I will be more than happy to discuss and explain!