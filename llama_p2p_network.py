import zmq
import time
import random
import json
import threading
import argparse
import logging
import hashlib
import os
import secrets
from llama_cpp import Llama
import pynng
from collections import deque, defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LlamaP2PNode:
    def __init__(self, model_path, port, known_peers=None, cache_size=100, secret_key=None):
        self.model = Llama(model_path=model_path)
        self.model_hash = self.compute_model_hash(model_path)
        self.port = port
        self.peers = set(known_peers) if known_peers else set()
        self.node_id = hashlib.sha256(f"{port}_{random.randint(1, 1000000)}".encode()).hexdigest()[:10]
        self.context = zmq.Context()
        self.gossip_socket = self.context.socket(zmq.PUB)
        self.gossip_socket.bind(f"tcp://*:{port}")
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        self.request_socket = pynng.Req0()
        self.reply_socket = pynng.Rep0()
        self.reply_socket.listen(f"tcp://0.0.0.0:{port+1}")
        self.cache = {}
        self.cache_queue = deque(maxlen=cache_size)
        self.lock = threading.Lock()
        self.active = True
        self.secret_key = secret_key or secrets.token_hex(16)
        self.peer_performance = defaultdict(lambda: {"success": 0, "failure": 0, "avg_time": 0})

    def start(self):
        for peer in self.peers:
            self.connect_to_peer(peer)

        threading.Thread(target=self.gossip_loop, daemon=True).start()
        threading.Thread(target=self.listen_loop, daemon=True).start()
        threading.Thread(target=self.handle_requests, daemon=True).start()

        logging.info(f"Node {self.node_id} started on port {self.port}")
        self.cli_loop()

    def connect_to_peer(self, peer):
        try:
            self.subscriber.connect(f"tcp://{peer}")
            logging.info(f"Connected to peer: {peer}")
        except zmq.ZMQError as e:
            logging.error(f"Failed to connect to peer {peer}: {e}")

    def gossip_loop(self):
        while self.active:
            message = {
                "type": "gossip",
                "node_id": self.node_id,
                "port": self.port,
                "peers": list(self.peers),
                "model_hash": self.model_hash
            }
            self.gossip_socket.send_json(message)
            time.sleep(5)

    def listen_loop(self):
        while self.active:
            try:
                message = self.subscriber.recv_json(flags=zmq.NOBLOCK)
                if message["type"] == "gossip":
                    new_peer = f"{message['node_id']}:{message['port']}"
                    if new_peer not in self.peers and new_peer != f"{self.node_id}:{self.port}":
                        logging.info(f"Discovered new peer: {new_peer}")
                        self.peers.add(new_peer)
                        self.connect_to_peer(new_peer)
                    if message["model_hash"] != self.model_hash:
                        logging.warning(f"Peer {new_peer} has a different model hash")
            except zmq.ZMQError:
                time.sleep(0.1)

    def handle_requests(self):
        while self.active:
            try:
                msg = self.reply_socket.recv(timeout=100)
                request = json.loads(msg.decode())
                if request["type"] == "inference" and request.get("secret_key") == self.secret_key:
                    result = self.cached_inference(request["prompt"])
                    response = {"result": result}
                    self.reply_socket.send(json.dumps(response).encode())
                else:
                    self.reply_socket.send(json.dumps({"error": "Unauthorized"}).encode())
            except pynng.Timeout:
                continue
            except Exception as e:
                logging.error(f"Error handling request: {e}")

    def cli_loop(self):
        while self.active:
            command = input("Enter command (infer/peers/cache/performance/exit): ")
            if command == "infer":
                prompt = input("Enter prompt: ")
                result = self.distributed_inference(prompt)
                print(f"Result: {result}")
            elif command == "peers":
                print(f"Known peers: {self.peers}")
            elif command == "cache":
                print(f"Cache size: {len(self.cache)}")
                print(f"Cache items: {list(self.cache.keys())}")
            elif command == "performance":
                self.print_performance_stats()
            elif command == "exit":
                self.shutdown()
                break
            else:
                print("Unknown command")

    def cached_inference(self, prompt):
        with self.lock:
            if prompt in self.cache:
                return self.cache[prompt]
            
            result = self.model(prompt, max_tokens=100)["choices"][0]["text"]
            self.cache[prompt] = result
            self.cache_queue.append(prompt)
            
            if len(self.cache) > self.cache_queue.maxlen:
                oldest = self.cache_queue.popleft()
                del self.cache[oldest]
            
            return result

    def distributed_inference(self, prompt):
        if not self.peers:
            return self.cached_inference(prompt)
        
        peer = self.select_peer()
        try:
            start_time = time.time()
            with pynng.Req0() as s:
                s.dial(f"tcp://{peer}")
                s.send(json.dumps({"type": "inference", "prompt": prompt, "secret_key": self.secret_key}).encode())
                response = s.recv()
            elapsed_time = time.time() - start_time
            result = json.loads(response.decode())["result"]
            self.update_peer_performance(peer, True, elapsed_time)
            return result
        except Exception as e:
            logging.error(f"Error communicating with peer {peer}: {e}")
            self.update_peer_performance(peer, False)
            self.peers.remove(peer)
            return self.distributed_inference(prompt)

    def select_peer(self):
        if not self.peer_performance:
            return random.choice(list(self.peers))
        return max(self.peer_performance, key=lambda x: self.peer_performance[x]["success"] / (self.peer_performance[x]["success"] + self.peer_performance[x]["failure"] + 1))

    def update_peer_performance(self, peer, success, elapsed_time=None):
        with self.lock:
            if success:
                self.peer_performance[peer]["success"] += 1
                if elapsed_time:
                    self.peer_performance[peer]["avg_time"] = (self.peer_performance[peer]["avg_time"] * (self.peer_performance[peer]["success"] - 1) + elapsed_time) / self.peer_performance[peer]["success"]
            else:
                self.peer_performance[peer]["failure"] += 1

    def print_performance_stats(self):
        for peer, stats in self.peer_performance.items():
            print(f"Peer {peer}:")
            print(f"  Success: {stats['success']}")
            print(f"  Failure: {stats['failure']}")
            print(f"  Avg Time: {stats['avg_time']:.2f}s")

    def compute_model_hash(self, model_path):
        hasher = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def shutdown(self):
        logging.info(f"Shutting down node {self.node_id}")
        self.active = False
        self.gossip_socket.close()
        self.subscriber.close()
        self.reply_socket.close()
        self.context.term()

def main():
    parser = argparse.ArgumentParser(description="Llama P2P Node")
    parser.add_argument("--model", required=True, help="Path to the GGUF model file")
    parser.add_argument("--port", type=int, required=True, help="Port to run the node on")
    parser.add_argument("--peers", nargs="*", help="Known peer addresses")
    parser.add_argument("--cache-size", type=int, default=100, help="Size of the inference cache")
    parser.add_argument("--secret-key", help="Secret key for node authentication")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        logging.error(f"Model file not found: {args.model}")
        return

    node = LlamaP2PNode(args.model, args.port, args.peers, args.cache_size, args.secret_key)
    try:
        node.start()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        node.shutdown()

if __name__ == "__main__":
    main()
