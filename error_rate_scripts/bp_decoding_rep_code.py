from tqdm import tqdm
import joblib as jl

import numpy as np
import matplotlib.pyplot as plt

from ldpc import BpDecoder


def rep_code(d: int):
    code = np.zeros((d - 1, d), dtype="u1")
    for i in range(d - 1):
        code[i][i] = 1
        code[i][i + 1] = 1
    return code

def get_wer(pcm, num_qubit, p, simulation_shots):
    errors = np.random.binomial(1, p, (simulation_shots, num_qubit))
    syndrome = (errors @ pcm.T) % 2
    dec = BpDecoder(
            pcm,#the parity check matrix
            error_channel=[p for _ in range(num_qubit)],
            max_iter=50,
            # bp_method="minimum_sum",
            # ms_scaling_factor=0.625, #min sum scaling factor
        )
    
    word_error = np.zeros(simulation_shots, dtype=bool)

    for i in range(simulation_shots):
        # print("#################")
        # print(f"error is {errors[i]}")
        s = syndrome[i]
        e_corrected = dec.decode(s)
        # print(f"predicted error {e_corrected}")
        net_coorectn = (errors[i] + e_corrected)%2
        word_error[i] = ''.join(map(str, net_coorectn)) != '0' * num_qubit
        word_error[i] = ''.join(map(str, e_corrected)) != ''.join(map(str, errors[i]))
        # print(f"word error {word_error[i]}")
    # print(f"net wer: {word_error}")
    return np.mean(word_error)


if __name__ == "__main__":
    distances = [5,9]
    noise_prob = np.array([0.1, 0.2, 0.3, 0.4])#np.arange(0.1,0.9,0.1)
    # samples to stimate WER
    simulation_shots = 500
    results = np.zeros((len(distances),len(noise_prob)))
    # results = np.empty((len(distances), len(noise_prob), simulation_shots))

    for i, d in enumerate(distances):
        num_qubit = d
        pcm = rep_code(d)
        results[i,:] = jl.Parallel(n_jobs=-1,backend="multiprocessing")(
                            jl.delayed(get_wer)(
                                pcm,
                                num_qubit,
                                p,
                                simulation_shots
                            )
                            for p in tqdm(noise_prob)
        )

    print(results)

    plt.figure(figsize=(10,6))

    for i, d in enumerate(distances):
        plt.plot(noise_prob, results[i], marker="o", label=f"distance: {d}")
    
    plt.legend()

    plt.savefig('data/bp_rep.png',dpi = 300)