from time import sleep
from json import dumps
from kafka import KafkaProducer

service_ip = "172.24.217.34:9092"
producer = KafkaProducer(bootstrap_servers=[service_ip],
                         value_serializer=lambda x: 
                         dumps(x).encode('utf-8'))

for e in range(5):
    data = {'number' : e}
    producer.send('my_test', value=data)
    sleep(5)