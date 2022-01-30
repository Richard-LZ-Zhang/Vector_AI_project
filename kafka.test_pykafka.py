from pykafka import KafkaClient
from pykafka.utils.compat import Empty
import queue
from datetime import datetime
import uuid
import time
import os

service_ip = "127.0.0.1:2186" # "127.0.0.1:9092" # "172.24.217.34:9092"
client = KafkaClient(hosts=service_ip)

print(client.topics)
topic = client.topics[b'my_topic']

# with topic.get_sync_producer() as producer:
#     # message = bytes("message",'UTF-8')
#     # producer.produce(message)
#     for i in range(5):
#         message = 'test another host message ' + str(i ** 4)
#         producer.produce(message.encode("utf-8"))

# with topic.get_producer(delivery_reports=True) as producer:
#     count = 0
#     while count < 1000:
#         count += 1
#         producer.produce(bytes("message "+str(count),'UTF-8'))
#         if count % 10 ** 2 == 0:  # adjust this or bring lots of RAM ;)
#             while True:
#                 try:
#                     msg, exc = producer.get_delivery_report(block=False)
#                     if exc is not None:
#                         print('Failed to deliver msg {}: {}'.format(
#                             msg.partition_key, repr(exc)))
#                     else:
#                         print('Successfully delivered msg {}'.format(
#                         msg.partition_key))
#                 except Queue.Empty:
#                     break

with topic.get_producer(
    min_queued_messages=1, max_queued_messages=1, delivery_reports=True
) as producer:
    partition_key = str.encode(str(uuid.uuid4()))
    for i in range(10):
        test_message = f"test message {str(i)} {datetime.now()}"
        producer.produce(str.encode(test_message), partition_key=partition_key)

        msg, exc = producer.get_delivery_report(block=True)
        if exc is not None:
            print("Failed to deliver msg {}: {}".format(msg.partition_key, repr(exc)))
        else:
            print("Successfully delivered msg {}".format(msg.partition_key))

    print("waiting for all messages to be written")
    producer._wait_all()