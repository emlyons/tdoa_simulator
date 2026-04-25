import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erf
from scipy.stats import beta


def _add_sigma(n_sigma, sigma, S, cdf):
    x = n_sigma * sigma
    y = np.interp(x, S, cdf)

    # vertical from bottom to curve and horizontal from right side to curve
    plt.plot([x, x], [0, y], 'k--', linewidth=1, label=f'{n_sigma}σ Threshold {n_sigma*sigma:.3f} : {erf((n_sigma*sigma) / (sigma * np.sqrt(2))):.3f}%')
    plt.plot([x, S.max()], [y, y], 'k--', linewidth=1)
    plt.scatter([x], [y], color='k')
    plt.annotate(f'{n_sigma}σ', xy=(x, y), xytext=(x + 0.2, min(1.0, y + 0.03)),
                arrowprops=dict(arrowstyle='->', lw=0.5))
    
def _add_confidence_bounds(sorted_data, cdf, alpha=0.05):
    """Add a DKW (Dvoretzky–Kiefer–Wolfowitz) uniform band around the empirical CDF.

    The band is Fn(x) +/- epsilon where epsilon = sqrt((1/(2n)) * ln(2/alpha)).
    This gives a simultaneous (uniform) confidence band for the true CDF with
    coverage at least 1-alpha.
    """
    n = len(sorted_data)
    if n <= 0:
        return

    # DKW epsilon (uniform band half-width)
    epsilon = np.sqrt((1.0 / (2.0 * n)) * np.log(2.0 / alpha))

    lower = np.clip(cdf - epsilon, 0.0, 1.0)
    upper = np.clip(cdf + epsilon, 0.0, 1.0)
    plt.fill_between(sorted_data, lower, upper, color="gray", alpha=0.3, label=f"{100*(1-alpha):.1f}% DKW uniform band")


def _plot_theoretical_cdf(sigma, size=100):
    S = np.linspace(0, 4*sigma, size)
    theoretical_cdf = erf((S) / (sigma * np.sqrt(2)))
    plt.plot(S, theoretical_cdf, 'r-', linewidth=2, label='Theoretical Gaussian CDF')
    _add_sigma(1, sigma, S, theoretical_cdf)
    _add_sigma(2, sigma, S, theoretical_cdf)
    _add_sigma(3, sigma, S, theoretical_cdf)


def _plot_empirical_cdf(data, sensor_id):
    mu = np.mean(data)
    sorted_data = np.sort(np.abs(data - mu))  # Center data around mean for better visualization
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=f'Empirical Sensor {sensor_id} CDF')
    _add_confidence_bounds(sorted_data, cdf)


def plot_cdf(data, sensor_id, sigma, file_name=None):
    plt.figure()
    _plot_theoretical_cdf(sigma)
    _plot_empirical_cdf(data, sensor_id)
    plt.title(f'Sensor {sensor_id} CDF')
    plt.ylabel('CDF')
    plt.xlabel('Value')
    plt.legend()
    plt.show()
    if file_name:
        plt.savefig(file_name)

    
def plot_toa_distribution(toa, file_name):
    pulse_period = np.diff(toa)
    pp_mean = np.mean(pulse_period)
    pp_std = np.std(pulse_period)

    # Histogram of pulse periods
    plt.figure()
    plt.hist(pulse_period, label=f"pulse period (s)", bins=20)
    
    # Normal Distribution Fitting
    x = np.linspace(np.min(pulse_period), np.max(pulse_period), 200)
    sigma = pp_std if pp_std > 0 else 1e-8
    pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - pp_mean) / sigma) ** 2)

    pdf = pdf / np.max(pdf) * np.max(np.histogram(pulse_period, bins=20)[0])
    # scale pdf to match histogram counts
    plt.plot(x, pdf, "r-", linewidth=2, label="Gaussian fit")
    plt.axvline(pp_mean, color="k", linestyle="--", linewidth=1, label="Mean")
    plt.text(0.95, 0.95, f"mean={pp_mean:.6f}\nstd={pp_std:.6f}", transform=plt.gca().transAxes,
                ha="right", va="top", fontsize=8)
    plt.xlabel("Pulse Period (s)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Arrival Jitter\nfor {} samples".format(len(pulse_period)))
    plt.legend()

    if file_name:
        plt.savefig(file_name)