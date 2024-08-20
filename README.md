# llama-p2p


## How It Works

1. **Node Discovery**: Nodes use a gossip protocol to discover and connect to other nodes in the network.

2. **Distributed Inference**: When a node receives an inference request, it can either process it locally or forward it to another node based on load balancing criteria.

3. **Caching**: Each node maintains a local cache of recent inference results to improve performance.

4. **Load Balancing**: The system tracks the performance of each peer and uses this information to distribute workload efficiently.

5. **Model Consistency**: Nodes check the hash of their model file to ensure consistency across the network.

## Limitations and Future Work

- The current implementation uses a simple load balancing strategy. More advanced strategies could be implemented.
- Security is basic. In a production environment, more robust security measures should be implemented.
- The system currently assumes all nodes have the same model. Future versions could support heterogeneous model deployments.
- There's no persistent storage of the network state or inference results between restarts.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
