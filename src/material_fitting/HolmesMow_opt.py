import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

sns.set('paper')

def HolmesMow(E, beta, v, stress, stretch, I_1, I_2, I_3):
    mu = E / (2 * (1 + v))
    lame2 = E * v / ((1 + v) * (1 - 2 * v))
    J = stretch
    q = beta / (lame2 + 2 * mu) * \
        ((2 * mu - lame2) * (I_1 - 3) + lame2 * (I_2 - 3) - (lame2 + 2 * mu) * np.log(J ** 2))
    s = 1.0 / (2 * J) * np.exp(q) * \
        ((2 * mu + lame2 * (I_1 - 1)) * stretch ** 2 - lame2 * stretch ** 4 - (lame2 + 2 * mu))
    return s

def obj(x, stress, stretch, v, I_1, I_2, I_3):
    E = x[0]
    beta = x[1]
    stress_pred = HolmesMow(E, beta, v, stress, stretch, I_1, I_2, I_3)
    return np.sqrt(np.linalg.norm(stress_pred - stress))

def main(config):
    with open(config) as user:
        values = yaml.load(user, Loader=yaml.FullLoader)

    df = {'H': [],
          'E': [],
          'v': [],
          'beta': [],
          'rms error': []}
    fig, ax = plt.subplots(1, len(values['stress']), sharey=True)
    for i, (stress, strain, v) in enumerate(zip(values['stress'],
                                                values['strain'],
                                                values['v'])):
        stress = np.array(stress)
        strain = np.array(strain)
        H = np.min(stress[1:] / strain[1:]) 
        E = H * (1 - 2 * v) * (1 + v) / (1 - v)
        stretch = strain + 1.0

        I_1 = stretch ** 2 + 2
        I_2 = 0.5 * (I_1 ** 2 - (stretch ** 4 + 2))
        I_3 = stretch ** 2

        x = [E, 1.0]
        bounds = ((0.5 * E, None),
                  (0.0, None))
        res = minimize(obj, x,
                       args = (stress,
                               stretch,
                               v,
                               I_1,
                               I_2,
                               I_3),
                       bounds = bounds,
                       method='SLSQP')

        df['H'].append(res.x[0] * (1 - 2 * v) * (1 + v) / (1 - v))
        df['E'].append(res.x[0])
        df['v'].append(v)
        df['beta'].append(res.x[1])
        df['rms error'].append(res.fun / stress.size)
        ax[i].plot(strain, stress, 'o', label='Measured')
        ax[i].plot(strain, HolmesMow(*res.x, v, stress, stretch, I_1, I_2, I_3), label='Predicted')
        #ax[i].set_ylabel('Stress (MPa)')
        #ax[i].set_xlabel('Strain')
        #ax[i].legend()

    plt.show()
    df = pd.DataFrame(df, index=values['depth'])
    df.plot.bar(subplots=True)
    plt.show()
    print(df)
    df.to_excel("holme-mow-solid.xlsx")


if __name__ == '__main__':
    main(sys.argv[-1])
