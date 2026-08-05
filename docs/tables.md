# Reward Shaping Tables

```latex
\begin{table}[h]
    \centering
    \scriptsize
    \renewcommand{\arraystretch}{1.15}
    \setlength{\tabcolsep}{4pt}

    \begin{tabularx}{\linewidth}{@{}%
        >{\raggedright\arraybackslash}p{0.24\linewidth}
        >{\centering\arraybackslash}X
        >{\centering\arraybackslash}p{0.13\linewidth}
    @{}}
    \toprule
    \textbf{Reward Shaping Quadruped Positive Reward Term} & \textbf{Definition} & \textbf{Scale} \\
    \midrule

    Linear velocity tracking &
    $\displaystyle
    r_{\text{track-lin}} =
    \exp\!\left(
    -\frac{
    \|\mathbf{v}^{\text{cmd}}_{xy}-\mathbf{v}^{\text{base}}_{xy}\|_2^2
    +2(v_z^{\text{base}})^2
    }{0.25}
    \right)
    $ &
    $\displaystyle w_{\text{track-lin}} = 4.0$ \\

    Angular velocity tracking &
    $\displaystyle
    r_{\text{track-ang}} =
    \exp\!\left(
    -\frac{
    (\omega_z^{\text{cmd}}-\omega_z^{\text{base}})^2
    +0.05\|\boldsymbol{\omega}^{\text{base}}_{xy}\|_2^2
    }{0.25}
    \right)
    $ &
    $\displaystyle w_{\text{track-ang}} = 2.5$ \\

    Forward linear velocity reward &
    $\displaystyle
    r_{\text{fwd-lin}}=
    \begin{cases}
    \operatorname{clip}\!\left(
    (\mathbf{v}^{\text{base}}_{xy})^\top
    \frac{\mathbf{v}^{\text{cmd}}_{xy}}{\|\mathbf{v}^{\text{cmd}}_{xy}\|_2},
    \,0,\,
    \|\mathbf{v}^{\text{cmd}}_{xy}\|_2
    \right),
    & \|\mathbf{v}^{\text{cmd}}_{xy}\|_2 > 0.1, \\[4pt]
    0, & \text{otherwise,}
    \end{cases}
    $ &
    $\displaystyle w_{\text{fwd-lin}} = 6.0$ \\

    Forward yaw-rate reward &
    $\displaystyle
    r_{\text{fwd-ang}}=
    \begin{cases}
    \operatorname{clip}\!\left(
    \omega_z^{\text{base}}\,\operatorname{sign}(\omega_z^{\text{cmd}}),
    \,0,\,
    |\omega_z^{\text{cmd}}|
    \right),
    & |\omega_z^{\text{cmd}}| > 0.1, \\[4pt]
    0, & \text{otherwise,}
    \end{cases}
    $ &
    $\displaystyle w_{\text{fwd-ang}} = 1.0$ \\

    Alive bonus &
    $\displaystyle
    r_{\text{alive}} = 1
    $ &
    $\displaystyle w_{\text{alive}} = 0.0$ \\

    Variable posture reward &
    $\displaystyle
    r_{\text{pose}}=
    \exp\!\left(
    -\frac{1}{12}\sum_{j=1}^{12}
    \frac{(q_j-q_{j,0})^2}{\sigma_j(c)^2}
    \right)
    $ &
    $\displaystyle w_{\text{pose}} = 0.42$ \\

    Base height tracking &
    $\displaystyle
    r_{\text{height}}=
    \exp\!\left(
    -\frac{(z^{\text{base}}-0.27)^2}{0.01}
    \right)
    $ &
    $\displaystyle w_{\text{height}} = 1.0$ \\

    Feet air-time reward &
    $\displaystyle
    r_{\text{air}}=
    \chi\,
    \max\!\left(T_{\text{air}}-\left|t_{\text{mode}}-T_{\text{air}}\right|,\,0\right),
    \qquad
    T_{\text{air}}=0.245
    $ &
    $\displaystyle w_{\text{air}} = 0.75$ \\

    Scheduled diagonal gait reward &
    $\displaystyle
    r_{\text{gait}}=
    \chi\,
    \max\!\left(
    2\left(
    \frac{1}{4}\sum_{i=1}^{4}\mathbbold{1}[I_i=S_i(\phi)]
    -\frac{1}{2}
    \right),\,0
    \right)
    $ &
    $\displaystyle w_{\text{gait}} = 1.35$ \\

    \bottomrule
    \end{tabularx}
    \caption{Positive reward terms for the fully reward shaped quadruped velocity tracking task. Here $c=\|\mathbf{v}^{\text{cmd}}_{xy}\|_2+|\omega_z^{\text{cmd}}|$, $\chi=\mathbbold{1}[c>0.1]$, and $I_i=\mathbbold{1}[i\in\text{contact}]$. For the air-time term, $t_{\text{mode}}=\min_i(I_i t_i^{\text{contact}}+(1-I_i)t_i^{\text{air}})$ when exactly two feet are in contact, and $t_{\text{mode}}=0$ otherwise. For the gait term, $\phi=(t/0.52)\bmod 1$, $S_i(\phi)=\mathbbold{1}[((\phi+\delta_i)\bmod 1)<0.52]$, with $\delta_{\mathrm{FR}}=\delta_{\mathrm{RL}}=0$ and $\delta_{\mathrm{FL}}=\delta_{\mathrm{RR}}=0.5$.}
    \label{tab:full_reward_positive}
\end{table}

\begin{table}[h]
    \centering
    \scriptsize
    \renewcommand{\arraystretch}{1.15}
    \setlength{\tabcolsep}{4pt}

    \begin{tabularx}{\linewidth}{@{}%
        >{\raggedright\arraybackslash}p{0.24\linewidth}
        >{\centering\arraybackslash}X
        >{\centering\arraybackslash}p{0.13\linewidth}
    @{}}
    \toprule
    \textbf{Reward Shaping Quadruped Penalty Term} & \textbf{Definition} & \textbf{Scale} \\
    \midrule

    Lateral velocity penalty &
    $\displaystyle
    c_{\text{lat}}=
    \chi\,(v_y^{\text{base}})^2
    $ &
    $\displaystyle w_{\text{lat}} = -1.0$ \\

    Flat orientation penalty &
    $\displaystyle
    c_{\text{flat}} = \|\mathbf{g}^{\text{proj}}_{xy}\|_2^2
    $ &
    $\displaystyle w_{\text{flat}} = -0.7$ \\

    Body angular velocity penalty &
    $\displaystyle
    c_{\text{body-ang}}=
    \|\boldsymbol{\omega}^{\text{world}}_{xy}\|_2^2
    $ &
    $\displaystyle w_{\text{body-ang}} = -0.16$ \\

    Pitch tilt penalty &
    $\displaystyle
    c_{\text{pitch}}=
    \theta_{\text{pitch}}^2
    $ &
    $\displaystyle w_{\text{pitch}} = -2.0$ \\

    Angular momentum penalty &
    $\displaystyle
    c_{\text{angmom}}=
    \|\mathbf{h}_{\text{base}}\|_2^2
    $ &
    $\displaystyle w_{\text{angmom}} = -0.014$ \\

    Termination penalty &
    $\displaystyle
    c_{\text{term}}= \mathbbold{1}[\text{terminated}]
    $ &
    $\displaystyle w_{\text{term}} = -10.0$ \\

    Joint acceleration penalty &
    $\displaystyle
    c_{\text{joint-acc}}=
    \|\ddot{\mathbf{q}}\|_2^2
    $ &
    $\displaystyle w_{\text{joint-acc}} = -3\times 10^{-7}$ \\

    Joint limit penalty &
    $\displaystyle
    c_{\text{joint-lim}}=
    \sum_{j=1}^{12}
    \Bigl[
    \max(\underline q^{\text{soft}}_j-q_j,0)
    +
    \max(q_j-\overline q^{\text{soft}}_j,0)
    \Bigr]
    $ &
    $\displaystyle w_{\text{joint-lim}} = -1.0$ \\

    Action rate penalty &
    $\displaystyle
    c_{\text{act-rate}}=
    \|\mathbf{a}_t-\mathbf{a}_{t-1}\|_2^2
    $ &
    $\displaystyle w_{\text{act-rate}} = -0.045$ \\

    Bad two-foot contact penalty &
    $\displaystyle
    c_{\text{bad-2}}=
    \chi\,
    \mathbbold{1}[|C|=2]\,
    \mathbbold{1}\!\left[
    C\notin
    \{\{\mathrm{FL},\mathrm{RR}\},\{\mathrm{FR},\mathrm{RL}\}\}
    \right]
    $ &
    $\displaystyle w_{\text{bad-2}} = -0.7$ \\

    Feet clearance penalty &
    $\displaystyle
    c_{\text{clr}}=
    \chi
    \sum_{i=1}^{4}
    |z_i-z_{\text{tar}}|\,\|\mathbf{v}_{i,xy}^{\text{foot}}\|_2,
    \qquad
    z_{\text{tar}}=0.07
    $ &
    $\displaystyle w_{\text{clr}} = -1.0$ \\

    Feet slip penalty &
    $\displaystyle
    c_{\text{slip}}=
    \chi
    \sum_{i=1}^{4}
    \|\mathbf{v}_{i,xy}^{\text{foot}}\|_2^2\,
    \mathbbold{1}[i\in\text{contact}]
    $ &
    $\displaystyle w_{\text{slip}} = -0.12$ \\

    Soft landing penalty &
    $\displaystyle
    c_{\text{land}}=
    \chi
    \sum_{i=1}^{4}
    f_i^{\text{contact}}\,
    \mathbbold{1}[i\in\text{first-contact}]
    $ &
    $\displaystyle w_{\text{land}} = -2\times 10^{-4}$ \\

    \bottomrule
    \end{tabularx}
    \caption{Penalty terms for the fully reward shaped quadruped velocity tracking task. Here $C=\{i:I_i=1\}$ is the set of feet in contact.}
    \label{tab:full_reward_penalty}
\end{table}
```
