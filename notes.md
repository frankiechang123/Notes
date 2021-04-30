

# Docker

## Basics

### Docker Platform

- Isolation: many containers on a host 
- container as a unit of development

### Architechture

Client-server: Docker client to Docker daemon

Components:

- Docker daemon(server)
  - manages images, containers, networks, volumes
- Docker Client
  - `docker run` sends requests
- Docker registries
  - stores Docker images
  - Default: Docker Hub
  - `docker pull`: pulls images from registry
  - `docker push`: push to the registry
- 



### 概念

#### Image vs container:  

- 像类和对象

- Image: read-only template for creating a container
- Container: a runable instance of an image
  - isolated filesystem

#### 镜像分层

- base image
  - Eg: building images based on `Ubuntu` images
- if shared: copy on write

#### Container Volumes

- connect specific filesystem paths of the container back to the host machine

  - Two kinds:

    - Named volumes

      - ```shell
        docker volume create <volume_name>
        docker run -dp 3000:3000 -v <volume_name>:<file_path_in_container> <container_name>
        ```

    - Bind mounts

      - developers get to choose the locations on files

        ```shell
        docker run -dp 3000:3000 -v <file_path_host>:<file_path_in_container> <container_name>
        ```

### Multi-Container App

- networking

- use Docker Compose to define service via `.yml` files

  - ```yaml
    version: "3.7"
    
    services:
      app:
        image: node:12-alpine
        command: sh -c "yarn install && yarn run dev"
        ports:
          - 3000:3000
        working_dir: /app
        volumes:
          - ./:/app
        environment:
          MYSQL_HOST: mysql
          MYSQL_USER: root
          MYSQL_PASSWORD: secret
          MYSQL_DB: todos
    
      mysql:
        image: mysql:5.7
        volumes:
          - todo-mysql-data:/var/lib/mysql
        environment: 
          MYSQL_ROOT_PASSWORD: secret
          MYSQL_DATABASE: todos
    
    volumes:
      todo-mysql-data:
    ```

### Multistage Build

- Multiple `FROM` statement in dockerfile
- one stage--> `COPY`---> new stage --> ...
- reduction in image size

### DockerFile

- ```dockerfile
  # syntax=docker/dockerfile:1
  FROM ubuntu:18.04 #image
  COPY . /app #copy file from host curr dir
  RUN make /app #run when container is created
  CMD python /app/app.py #what to run when started
  ```

- [see examples](https://docs.docker.com/engine/reference/builder/#dockerfile-examples)

- `docker build` builds the image

#### Starting container

- `docker run`

- ```shell
   docker run -dp 3000:3000 getting-started
  ```

#### Replacing/Updating old container

```shell
#get ID
docker ps
docker stop <id>
docker rm <id>
```

#### Pushing the image

https://docs.docker.com/get-started/04_sharing_app/

```shell
docker push <name>
```



## Building Apps



#### Volumes

- Use Volume instead of bind mounts TODO

#### Kubernetes

- containers scheduled as pods

- workloads scheduled as deployments(groups of pods)

- YAML files:

  - describes all components and configurations

  - ```yaml
    apiVersion: apps/v1
    kind: Deployment #specifies kind
    metadata: 
      name: bb-demo
      namespace: default
    spec:
      replicas: 1 #number of pods
      selector: 
        matchLabels:
          bb: web
      template: #pod template starts
        metadata:
          labels:
            bb: web
        spec:
          containers: #containers in the pod
          - name: bb-site
            image: bulletinboard:1.0
        # pod template ends
    --- #objects are sperated by ---
    apiVersion: v1
    kind: Service
    metadata:
      name: bb-entrypoint
      namespace: default
    spec:
      type: NodePort
      selector:
        bb: web
      ports:
      - port: 8080
        targetPort: 8080
        nodePort: 30001
    ```

  - See https://docs.docker.com/get-started/kube-deploy/

  - 

#### Scale app as a Swarm







# Kubernetes

将容器分类组成“容器集”(Pods)

--> "Desired State Management"

<img src="/Users/yiyangzhou/Desktop/Stuff/Notes/images/components-of-kubernetes.svg" style="zoom:50%;" />

## Cluster Architechture

### Control Plane

- Manges worker nodes/pods
  - makes decisions: e.g. ensure `replicas` 
- Components
  - Kube-apiserver 
    - Communications between Control Plane and Nodes
  - etcd
  - Kube-scheduler
    - watches newly created pods
    - assign pods to node
  - Kube-controller-manager
    - runs the controller process
  - cloud-controller-manager
    - link cluster to cloud provider's API

### Node

- Identified by name; 两个Node不能有相同的name

##### Components

- kubelet
  - make sure containers are running in a pod
  - ensure normal pod termination process
    - Graceful node shutdown
      - Phase1: terminate regular pods
      - Phase2: terminate critical pods
- container runtime
  - e.g, docker
- Kube-proxy
  - maintains network rules

##### Node Status

- Addresses地址
- Conditions状况
- 容量与可分配： `capacity`, `allocatable`
- 信息: 版本信息

##### Node Controller节点控制器

- Assign a CIDR block to Node
- 保证节点控制器内的节点列表与服务商所提供的可用机器列表同步
- monitoring nodes' health
  - update Node status: e.g. `NodeReady` to `ConditionUnknown`
  - evict unreachable pods

##### Heartbeat心跳

- `NodeStatus`
- Lease object
  - lease object is updated every 10 seconds, independent from `NodeStatus` updates

## Kubernetes Objects

### Pods

- Smallest deployble units
- one or more containers with shared storage/network (usuallly one)
-  Pod templates: see examples above
  - included in workload resourses like `Deployment`, `Job`
- unique IP address

### Workload Resources

​		workload resources configure controllers that ensures the "desired state"

#### Deployments

Configures a controller. Provides updates for Pods and ReplicaSets

- upgrade/scale/undo

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
	labels:
		environment: test
	name: testdeploy
spec:
	#create a replica set
	replicas: 3
	selector:
		matchLabels:
			environment: test
	#defines how to update
	minReadySeconds: 10
	strategy:
		rollingUpdate:
			maxSurge: 1
			maxUnavalable: 0
		type: RollingUpdate
	
	template: #pod template
		metadata:
			labels:
				environment: test
		spec:
			containers:
				- image: nginx:1.17
					name: nginx
		
```

Pop updates/replacement:

- update the deployment configuration directly
- the controller will update the pods automatically

https://kubernetes.io/docs/concepts/workloads/controllers/deployment/

#### ReplicaSet

- wraps a groups of pods
- `replicas:<num>`

#### DeamonSet

- similar to Deployment. 
- Difference:
  - ensures all Nodes run a copy of pod
  - no `replica` field

#### Jobs

- create pods to do a job

```yaml
#use perl image to compute pi
apiVersion: batch/v1
kind: Job
metadata:
  name: pi
spec:
  template:
    spec:
      containers:
      - name: pi
        image: perl
        command: ["perl",  "-Mbignum=bpi", "-wle", "print bpi(2000)"]
      restartPolicy: Never #better debugging
  backoffLimit: 4
```

- 可以并发

### Service

保持IP address; *decoupling*

- In Deployments, IP may be different from time to time
- 前端不需要知道后端变化

```yaml
#a new Service object for any Pod with app=Myapp label
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  selector:
    app: MyApp
  ports:
    - protocol: TCP
      port: 80
      targetPort: 9376 #redirect to targetport
```





## CLI

```shell
#create deployment
kubectl create -f example.yaml
#check status
kubectl get <kind>
kubectl get pods
#scaling up or down
kubectl replace -f example.yaml
#OR
kubectl scal - -replica=<num> -f example.yaml

#delete pod
kubectl delete pod <pod_id>

#get description
kubectl describe <kind> <name>

```



# Redis

Open source, in-memory data structure store

Command line: redis-cli

## Redis data types

### Redis Keys

- Binary safe: anything can be a key
-  建议：`object-type:id:field`
- Maximum key size: 512MB

### Redis String



### Redis Lists

- Use cases:

  - lasted updates by a user
  - communication between processes

  `rpush`, `lpush`, `lrange`, `lpop`, `rpop`, 

  blocking version of pop: `brpop`, `blpop`

```shell
rpush mylist A
lpush mylist B

```



### Capped Lists

 only store the latest items

```shell
LPUSH mylist <some element>
LTRIM mylist 0 999 #discard elements exceeding a limit
```



#### Automatic Creation and removal of keys

1. for an aggregate data type, if key not exists, it's created.
2. If value becomes empty, key is destroyed
3. Read-only commands(eg `LLEN`) treats empty key as empty



### Redis Hashes

```shell
> hmset user:1000 username antirez birthyear 1977 verified 1
OK
> hget user:1000 username
"antirez"
> hget user:1000 birthyear
"1977"
> hgetall user:1000
1) "username"
2) "antirez"
3) "birthyear"
4) "1977"
5) "verified"
6) "1"
```

`HMSET`: sets multiple fields

`HGET`: retrieves a single field

`HMGET`: retrives an array of values

`HINCRBY`: increment fields

https://redis.io/commands#hash



### Redis Set

`sadd`

`smembers`

`sismember`



### Redis Sorted Sets

`zadd`

`zrange`

### Redis Bitmaps

`SETBIT`

`GETBIT`

`BITOP`: bit-wise operations

`BITCOUNT`



## Redis Architechture

复制Replication: 主从复制Master-Slave Architechture

- 对master读写
- 对slave只读
- master offers copies to slaves
- 默认异步复制， 同步复制： use `WAIT`
- 

哨兵模式 Redis Sentinel

- 基于主从模式
- 原理
  - Sentinel每一秒向Master,slave, other sentinel发送 `ping` 命令
    - if timeout: 被标记为主观下线
    - 如果足够多sentinel确认主观下线： 被标记为客观下线
      - 投票选出新主节点

集群模式 Redis Cluster

- 无中心，每个节点都可保存全部状态
- 多个master
  - 可以分哈希槽hash slot
  - Failover： 所属slaves 投票

### Pipelining

### Transactions

### Partitioning



### 数据量

https://www.cnblogs.com/yilezhu/p/9941208.html

- “天处理近1亿包裹数据，日均调用量80亿次”
- Get/set 平均耗时 200-600us
- 最大支持1000并发

From Redis doc: 

- "deliver 1 million requests per second"
- "can hadle up to 2^32 keys"

Memory Footprint内存占用

- empty instance: 3MB
- 1M small keys, small value pairs: 85MB
- 1M keys,Large objects(5 fields): 160MB

Memory Optimization: https://redis.io/topics/memory-optimization



# Apache Hive

- a data warehouse.
- Standard SQL functionality
  - 3 choices of runtime:
    - Apache Hadoop MapReduce, Apache Tez or Apache Spark
  - data warehousing tasks:
    - Extract/transform/load
    - reporting 
    - data analysis

SQL to MapReduce: https://tech.meituan.com/2014/02/12/hive-sql-to-mapreduce.html

Example see tutorial



Talend.com: 100,000 queries/hour.

# Apache RocketMQ

Alternatives:

- Kafka
  - 把收到的所有信息写入disk 文件末尾
    - 分区会产生过多文件
  - 消息批量发给broker，减少网络IO

### 消息模型 Message Model

- producer

  - 把产生的信息发送到broker服务器

  - 多种发送方式（同步`send(msg)`，异步`send(msg,callback)`，顺序，单向`sendOneway(msg)`）

  - Producer Group: 同类集合

  - ```java
    DefaultMQProducer producer = new DefaultMQProducer("group name");
    ```

- consumer

  - 从broker server拉取信息，提供给应用程序

  - 分类

    - Pull Consumer： 需要pull from broker
    - Push Consumer：broker 实时push

  - consumer group: 同类集合

  - ```java
    DefaultMQPushConsumer consumer = new DefaultMQPushConsumer("group name");
    ```

- Broker

  - 中转： 储存信息，转发信息

### 消费模型

- 集群消费Clustering
  - 相同Consumer Group每个Consumer平均分摊信息
- 广播消费Broadcasting
  - 相同Consumer Group每个Consumer接收全量信息

### 其他概念

- 主题Topic
  - 一类信息的集合
- Name Server
  - broker registry
  - provide routing info for brokers
- 消息 Message
  - must have a topic
  - has a unique Message ID
- 标签 Tag
  - 区分同一主题下不同类型的消息
- 信息顺序
  - 全局顺序
  - 分区顺序
- 消息过滤
  - at broker server
  - 根据Tag进行过滤
- RocketMQ支持回溯消费
- 定时消费
  - 延迟投递给topic
- 消息重试
  - Consumer失败
- 消息重投
  - 同步消息失败会重投
  - 异步消息会重试
- 流量控制Flow Control
  - consumption capacity may reach bottleneck
- 死信队列Dead Letter Queue
  - 无法正常处理的消息



## Architechture

- Broker Cluster
  - Master-Slave structure
  - 消息储存结构
    - Commit Log：储存信息主体
    - ConsumeQueue：储存消息索引， by topic
    - IndexFile： 索引
- Producer Cluster
- Consumer Cluster
- NameServer Cluster

流程

- NameServer
- Broker， 长连接NameServer，注册，定时发送心跳/routing info
- Broker创建Topic，指定topic储存
- Producer/Consumer 长连接 NameServer，得到信息，与Broker建立连接
  - Consumer 向broker发送心跳包
  - Consumer Push/Pull
    - 默认：	pushConsumer： implemented by pull+长轮询
  - 



### 事物信息

1. Producer Send Half Msg To Broker
2. Broker响应
3. 如果成功，执行本地事物
4. Commit or Rollback （内部由Op消息实现）
   1. If commit： Send Message to consumer （若超时：不断重试）
   2. If Rollback: Delete
   3. 如果一直pending， broker需要发起回查

4之前对用户不可见

### 顺序消费

将有相似特征（订单id）的消息发送到同一Message Queue



### 消息重复

RocketMQ 不处理消息重复（大多由网络延迟引起）

可交由客户端处理

### 信息查询

两种方法

- Message Id(16Bytes: 储存主机地址，Commit Log Offset)
- Message Key
  - use indexFile
    - IndexFile entry: Key Hash/Commit Log Offset/Timestamp/NextIndex Offset

### Code Examples

https://github.com/apache/rocketmq/blob/master/docs/cn/RocketMQ_Example.md

### 客户端配置

- Producer, Consumer都是客户端



### 实际应用表现

https://segmentfault.com/a/1190000022962914

单机吞吐量：10万级

Topic 数量可以达到几百几千，吞吐量只会有小幅下降



# Pytorch

### Working with data

- `Dataset`

- `DataLoader` wraps an iterable around a `Dataset`
  - `batch_size`: batch size of the iterator

### Model

```python
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() #Flatten() convert multidemention to 1D
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(), #activation function
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10), # 3 layers
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
      
     #need a loss_fn, optimizer
    def train(dataloader, model, loss_fn, optimizer):
      pass
    def test(dataloader, model):
      pass

model = NeuralNetwork().to(device)
print(model)

X=torch.rand(1,28,28,device=device)
model(X)
```

### Tensors

Attributes

- `tensor.shape`
- `tensor.dtype`
- `tensor.device`

```python
#from data
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
#from numpy
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
#from tensor with same (shape,datatype)
x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float)
```

### 关于Pytorch性能

https://github.com/YellowOldOdd/SDBI

Batch变大可以提高Throughput ： 3 processes+Batch32： 1215.47 pic/s



Dynamic Batching： 开源项目SDBI， 将零碎的数据组合打包

# TensorRT

推理inference

compress/optimize neural network

combines layer/optimize kernel selection/...

- Layer & Tenser Fusion
  - 横向/纵向合并
  - 使用更少CUDA核心
- Weight& Activation Precision Calibration数据精度校准
  - 降低数据精度			---> 改善内存，延迟

To consume PyTorch models:

https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorrt/

- Use `Torch.onnx`
- `tensort.OnnxParser` to parse a Onnx model into TensorRT
- Apply optimization and generate an engine (The engine can be serialized for reuse )
- perform inference

Concepts:

- Builder: takes a network in TensorRT and gennerates an engine
- Engine: takes input data, perform inferences, emits inference output





## GPU

Goals

- Highly paralleled computations
- 

### ALU



Cuda in Nvidia GPU card: Compute unified device architechture



----------------



# RPC	

Remote Procedure Call

远程调用的问题

- Function ID： 用ID来确定哪个函数
- 序列化和反序列化：用于传参数
- 网络传输

# Thrift

an RPC framework



### `.thrift` file

```
```

### Generate sorce code from `.thrift`

```shell
thrift
thrift -r --gen py example.thrift
```

### Do RPC in client and server

https://thrift.apache.org/tutorial/py.html



## Layers

#### Transport

Client: `Transport`

```python
 # Make socket
  transport = TSocket.TSocket('localhost', 9090)

  # Buffering is critical. Raw sockets are very slow
  transport = TTransport.TBufferedTransport(transport)
  transport.open()
  transport.close()

```

Server: `ServerTransport`

```python
transport = TSocket.TServerSocket(host='127.0.0.1', port=9090)
```



#### Protocol

defines the scheme for serialization/deserialization (如何序列化和反序列化)

https://thrift.apache.org/docs/concepts.html

#### Processor

encapsulates the ability to read from input streams and write to output streams

```java
interface TProcessor {
    bool process(TProtocol in, TProtocol out) throws TException
}
```



#### Server

pulls everything together

1. 创建transport
2. 创建input/output protocols for the transport
3. processor
4. wait for connection

```python
server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
server.serve()
```



# Protobuf

serializing structured data  序列化

https://developers.google.com/protocol-buffers/docs/pythontutorial



