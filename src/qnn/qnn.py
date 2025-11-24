
class QNN:
    pass
    """
    Interface for CudaQNNs and QuantumQNNs
    """


def get_qnn(qnn_name, x_wires, num_layers, device='cpu'):
    import qnn.cuda_qnn as cuda_qnn
    assert 'cuda' in qnn_name.lower()
    return getattr(cuda_qnn, qnn_name)(
            num_wires=len(x_wires),
            num_layers=num_layers,
            device=device
        )

