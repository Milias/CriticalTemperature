from common import *


class DynamicDataPlot:
    def __init__(self, sys, N, mu_e):
        self.sys = sys
        self.N = N
        self.mu_e, self.mu_h = mu_e, self.sys.get_mu_h_ht(mu_e)

        self.u0_min, self.u1_min = None, None
        self.u0_max, self.u1_max = None, None
        self.u0, self.u1 = None, None
        self.du0, self.du1 = None, None

        self.img_left, self.img_right = None, None

        self.green = None
        self.green_colors = None

    def setImageObject(self, img_left, img_right):
        self.img_left, self.img_right = img_left, img_right

    def colorize(self):
        if self.green is None:
            return

        green_r, green_ph = abs(self.green), angle(conj(self.green))

        green_h = 0.5 + 0.5 * green_ph / pi
        green_s = 0.9 * ones_like(green_r)
        green_v = green_r / (1.0 + green_r)

        green_hsv = array([green_h, green_s, green_v]).T
        self.green_colors = matplotlib.colors.hsv_to_rgb(green_hsv)

    def changeFrame(self, u0_min, u0_max, u1_min, u1_max):
        self.u0_min = u0_min if u0_min > 1e-8 else (1e-8)
        self.u0_max = u0_max if u0_max < (1.0 - 1e-8) else (1.0 - 1e-8)
        self.u1_min = u1_min if u1_min > (-1.0 + 1e-8) else (-1.0 + 1e-8)
        self.u1_max = u1_max if u1_max < (1.0 - 1e-8) else (1.0 - 1e-8)

        self.u0, self.du0 = linspace(
            self.u0_min,
            self.u0_max,
            self.N,
            retstep=True,
        )

        self.u1, self.du1 = linspace(
            self.u1_min,
            self.u1_max,
            self.N,
            retstep=True,
        )

    def genData(self):
        if self.u0_max is None:
            return

        wk_vec = list(
            itertools.product(
                self.u1 / (1 - self.u1**2),
                self.u0 / (1 - self.u0),
            ))

        self.green = array(
            plasmon_green_ht_v(
                wk_vec,
                self.mu_e,
                self.mu_h,
                self.sys,
                1e-12,
            )).reshape((self.N, self.N)).T

    def updateImage(self):
        self.colorize()

        self.img_left.set_data(self.green_colors[::-1])
        self.img_right.set_data(self.green_colors[::-1, ::-1])

        self.img_left.set_extent((
            self.u0_min,
            self.u0_max,
            self.u1_min,
            self.u1_max,
        ))

        self.img_right.set_extent((
            -self.u0_max,
            self.u0_min,
            self.u1_min,
            self.u1_max,
        ))

    def update(self, ax):
        ax_lims = ax.viewLim
        x0, x1 = ax_lims.intervalx
        y0, y1 = ax_lims.intervaly

        self.changeFrame(x0, x1, y0, y1)
        self.genData()
        self.updateImage()

        ax.figure.canvas.draw_idle()


N_u = 1 << 9

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300  # K
sys = system_data(m_e, m_h, eps_r, T)

u0_max, u1_max = 0.9, 0.9

print('β^-1: %f eV' % (1 / sys.beta))

mu_e = 1 / sys.beta  # eV

dyn_plot = DynamicDataPlot(sys, N_u, mu_e)

fig = plt.figure(figsize=(16 * 1.2, 9 * 1.2))
ax = fig.subplots()

img1 = ax.imshow(
    zeros((N_u, N_u, 3)),
    aspect='auto',
    extent=(0, u0_max, -u1_max, u1_max),
)

img2 = ax.imshow(
    zeros((N_u, N_u, 3)),
    aspect='auto',
    extent=(-u0_max, 0, -u1_max, u1_max),
)

dyn_plot.setImageObject(img1, img2)

ax.set_autoscale_on(False)  # Otherwise, infinite loop

#ax.axis([-u0_max, u0_max, -u1_max, u1_max])
ax.axis([0, u0_max, -u1_max, u1_max])
plt.tight_layout()

ax.callbacks.connect('xlim_changed', dyn_plot.update)
ax.callbacks.connect('ylim_changed', dyn_plot.update)

dyn_plot.update(ax)

plt.show()
