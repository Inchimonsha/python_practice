import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

# np.random.seed(80)

def get_rand_r(lm, N):
    return np.random.exponential(scale=lm, size=N)

def show_hist(data, bins):
    plt.hist(data, bins=bins)
    plt.xticks(bins, rotation=45, ha="right")
    plt.xlabel("Интервалы")
    plt.ylabel("Частота")
    plt.title("Гистограмма частот")
    plt.tight_layout()
    plt.show()


def merge_intervals(observed_freq, expected_freq, min_count=5):
    """Объединяет интервалы с ожидаемым количеством меньше min_count."""
    merged_observed = []
    merged_expected = []

    current_observed = 0
    current_expected = 0

    for i in range(len(observed_freq)):
        current_observed += observed_freq[i]
        current_expected += expected_freq[i]

        # Если ожидаемая частота меньше min_count и это не последний интервал
        if current_expected < min_count and i < len(observed_freq) - 1:
            continue  # Продолжаем накапливать значения
        else:
            merged_observed.append(current_observed)
            merged_expected.append(current_expected)
            current_observed = 0
            current_expected = 0
    return np.array(merged_observed), np.array(merged_expected)

def test(x, s = 1, K=10, alpha=0.01, is_show_hist=False):
    N = len(x)
    T = np.max(x)
    # K = int(np.sqrt(N))
    # print(K)

    bin_width = T / K
    bins = np.linspace(0, T, K+1)
    observed_freq, _ = np.histogram(x, bins=bins)

    if is_show_hist: show_hist(observed_freq, bins)

    # x_a = (bins[:-1] + bins[1:]) / 2
    # x_v = np.sum(x_a * observed_freq) / N
    x_v = np.mean(x)
    lm = 1 / x_v

    n_T = []
    for i in range(0, len(bins)-1):
        P_i = np.exp(-lm * bins[i]) - np.exp(-lm * bins[i+1])
        n_T.append(N * P_i)

    n_Y = np.array(observed_freq)
    n_T = np.array(n_T)

    n_Y_merged, n_T_merged = merge_intervals(n_Y, n_T)

    Xi = np.sum((n_Y_merged - n_T_merged) ** 2 / (n_T_merged + 1e-10))

    r = len(n_T_merged) - (s + 1)
    p_val = 1 - chi2.cdf(Xi, r)
    res = ("Несущественные "
           "-> распределение согласующимися с теоретическим") if p_val > alpha \
        else ("Расхождения неслучайные "
              "-> избранный закон распределения отвергается")
    return res, p_val, r

N = 1000
lm = 20
x = get_rand_r(lm=lm, N=N)
K = 10

res, p_val, r = test(x, s=1, K=K, is_show_hist=False)
if res:
    print(res, p_val, r, sep='\n')