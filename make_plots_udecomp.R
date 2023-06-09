# ------------------------------------------------------------------------------
# MAKE PLOTS FOR UDECOMP PAPER
# ------------------------------------------------------------------------------

library(data.table)
library(ggplot2)
library(ggridges)
library(gridExtra)
library(mvtnorm)
library(RColorBrewer)

plot_dir = "/home/lisa-wm/Documents/1_work/1_research/repos/udecomp/figures/"
results_dir = c(
    "/home/lisa-wm/Documents/1_work/1_research/repos/udecomp/results"
)
metric_names = c("AU", "EU", "TU")
random_seeds = c("123", "456", "789")

aggregate_dts = function(dt_list) {
  dt_list = do.call(rbind, dt_list)
  dt_list[
    , mean_value := mean(value), by = list(pl, ds, case, metric)
    ][, sd_value := sd(value), by = list(pl, ds, case, metric)]
  dt_list$ds[dt_list$ds == "mnist_standard"] = "MNIST"
  dt_list$ds[dt_list$ds == "cifar10"] = "CIFAR10"
  dt_list
}
compute_binary_entropy = function(theta) {
    theta_both = c(theta, 1 - theta)
    - sum(theta_both * log2(theta_both))
}
compute_diversity = function(x) {
    x = x + 0.0001
    avg_x = colMeans(x)
    kld = sapply(
        seq_len(nrow(x)),
        function(x) sum(x * log2(x / avg_x))
    )
    mean(kld)
}

make_line_plot = function(
    dt, 
    xlab, 
    title, 
    col_width = 2, 
    bar_width = 1, 
    ax_factor = 1, 
    with_perf = TRUE
  ) {
  p = ggplot(dt, aes(x = case, y = mean_value)) +
    theme_classic()
  if (with_perf) {
    p = p +
      geom_col(
        unique(dt[metric %in% c("acc", "ece"), .(case, metric, mean_value)]),
        mapping = aes(x = case, y = mean_value / ax_factor, fill = metric),
        width = col_width,
        position = position_dodge2(
          width = 1, preserve = "single", padding = -0.5
        ),
        alpha = 0.7
      )
  }
  p = p +
    geom_line(
      dt[metric %in% c("tu", "au", "eu")], 
      mapping = aes(x = case, y = mean_value, col = metric)
    ) +
    geom_point(
      dt[metric %in% c("tu", "au", "eu")],
      mapping = aes(x = case, y = mean_value, col = metric)
    ) +
    geom_errorbar(
      unique(
        dt[
          metric %in% c("tu", "au", "eu"), .(case, metric, mean_value, sd_value)
        ]
      ),
      mapping = aes(
        ymin = mean_value - sd_value,
        ymax = mean_value + sd_value,
        col = metric
      ),
      # position = "jitter",
      width = bar_width
    ) +
    labs(x = xlab, y = "uncertainty", title = title) +
    scale_color_manual(
      "",
      labels = metric_names,
      values = c("cornflowerblue", "darkorange1", "black")
    ) +
    scale_fill_manual(
      "", labels = c("ACC", "ECE"), values = c("darkgray", "black")
    ) +
    scale_alpha_manual("", labels = NULL, values = c(0.6, 0.1)) +
    scale_x_continuous(breaks = unique(dt$case)) +
    scale_y_continuous(
      sec.axis = sec_axis(
        ~ . * ax_factor, name = "ACC / ECE"
      )
    ) +
    theme(legend.position = "right")
  p
}

make_density_plot = function(cases, experiment, pl, ds, utype, title, ncol) {
  vecs = lapply(
    cases,
    function(i) {
      vecs = lapply(
        random_seeds,
        function(j) {
          vec = fread(
            sprintf(
              "%s/%s/%s/%s/%s/%s/%s_full.csv",
              results_dir,
              experiment,
              pl,
              ds,
              j,
              i,
              utype
            ),
            drop = 1
          )
          vec[, rs := j]
        }
      )
      dt = do.call(rbind, vecs)
      dt[, case := as.numeric(i)]
    }
  )
  ggplot(do.call(rbind, vecs), aes(x = V2, col = rs)) +
    geom_density() +
    xlim(c(-0.1, 1)) +
    scale_y_continuous(breaks = NULL) +
    facet_wrap(~ case, scales = "free_y", ncol = ncol) +
    labs(x = toupper(utype), title = title) +
    theme_classic() +
    scale_color_grey(start = 0.2, end = 0.6) +
    theme(legend.position = "none")
}

make_tables_five = function(cases, experiment, pl, ds, utype, experiment_str) {
  vecs = lapply(
    cases,
    function(i) {
      vecs = lapply(
        random_seeds,
        function(j) {
          vec = fread(
            sprintf(
              "%s/%s/%s/%s/%s/%s/%s_five.csv",
              results_dir,
              experiment,
              pl,
              ds,
              j,
              i,
              utype
            ),
            drop = 1
          )
          list(mean(vec$V2), sd(vec$V2))
        }
      )
      list(
        mean(sapply(vecs, function(i) mean(i[[1]]))), 
        mean(sapply(vecs, function(i) mean(i[[2]])))
      )
    }
  )
  data.table(
    experiment = experiment_str,
    case = cases, 
    pl = pl,
    ds = ds,
    metric = utype,
    mean = round(sapply(vecs, function(i) i[[1]]), 4),
    sd = round(sapply(vecs, function(i) i[[2]]), 4)
  )
}

make_dataset_plots = function(dt, title, ncol) {
  ggplot(
    dt, 
    aes(x = x_1, y = x_2, col = y)
  ) +
    theme_classic() +
    geom_point() +
    labs(x = expression(x[1]), y = expression(x[2]), title = title) +
    scale_color_viridis_d("class", end = 0.9) +
    facet_wrap(~ case, ncol = ncol, scales = "free")
}

# PLOT UNCERTAINTY VS SAMPLE SIZE ----------------------------------------------

configs = expand.grid(
  c("laplace", "ensemble"),
  c("mnist_standard", "cifar10"),
  c("tu", "au", "eu")
)

fls = list.files(results_dir, pattern = "uvssamples.csv", recursive = TRUE)
dt_uvssamples = lapply(
  fls, function(i) unique(fread(paste(results_dir, i, sep = "/"), drop = 1))
)
dt_uvssamples = aggregate_dts(dt_uvssamples)

ggsave(
    paste0(plot_dir, "uvssamples_mnist_laplace.png"),
    make_line_plot(
      dt_uvssamples[pl == "laplace" & ds == "MNIST"], 
      "sample size in % of total",
      "MNIST, Laplace approximation",
      2,
      1,
      1
    ),
    height = 2,
    width = 5
)
ggsave(
  paste0(plot_dir, "uvssamples_mnist_laplace_eu_full.png"),
  make_density_plot(
    as.character(sort(unique(dt_uvssamples$case))),
    "samples",
    "laplace",
    "mnist_standard",
    "eu",
    "EU vs sample size in % (MNIST, Laplace approximation)",
    3
  ),
  height = 4,
  width = 6
)
ggsave(
  paste0(plot_dir, "uvsresolution_mnist_laplace_eu_10.png"),
  make_density_plot(
    "10",
    "resolution",
    "laplace",
    "mnist_standard",
    "eu",
    "EU vs image resolution in %",
    2
  ),
  height = 2,
  width = 5
)
ggsave(
    paste0(plot_dir, "uvssamples_mnist_ensemble.png"),
    make_line_plot(
      dt_uvssamples[pl == "ensemble" & ds == "MNIST"], 
      "sample size in % of total",
      "MNIST, deep ensemble",
      2,
      1,
      10
    ),
    height = 2,
    width = 5
)
ggsave(
  paste0(plot_dir, "uvssamples_mnist_ensemble_zoom.png"),
  make_line_plot(
    dt_uvssamples[pl == "ensemble" & ds == "MNIST"], 
    "sample size in % of total",
    "MNIST, deep ensemble",
    2,
    1,
    1
  ),
  height = 2,
  width = 5
)
ggsave(
    paste0(plot_dir, "uvssamples_cifar_laplace.png"),
    make_line_plot(
      dt_uvssamples[pl == "laplace" & ds == "CIFAR10"], 
      "sample size in % of total",
      "CIFAR10, Laplace approximation",
      2,
      1,
      1
    ),
    height = 2,
    width = 5
)
ggsave(
    paste0(plot_dir, "uvssamples_cifar_ensemble.png"),
    make_line_plot(
      dt_uvssamples[pl == "ensemble" & ds == "CIFAR10"], 
      "sample size in % of total",
      "CIFAR10, deep ensemble",
      2,
      1,
      1
    ),
    height = 2,
    width = 5
)

uvssamples_five = lapply(
  seq_len(nrow(configs)),
  function(i) {
    make_tables_five(
      sort(unique(dt_uvssamples$case)), 
      "samples", 
      configs[i, 1], 
      configs[i, 2], 
      configs[i, 3], 
      "sample size"
    )
  }
)
uvssamples_five = do.call(rbind, uvssamples_five)
uvssamples_five$ds = ifelse(
  uvssamples_five$ds == "mnist_standard", "MNIST", "CIFAR10"
)
uvssamples_five = uvssamples_five[
  dt_uvssamples[
    metric %in% c("tu", "au", "eu"), .(case, pl, ds, metric, mean_value)
  ], 
  on = list(case, pl, ds, metric)
  ][, metric := toupper(metric)
    ][, mean_value := round(mean_value, 4)]
uvssamples_five$pl = ifelse(
  uvssamples_five$pl == "ensemble", "deep ensemble", "Laplace approximation"
)
uvssamples_five = unique(uvssamples_five)
setorder(uvssamples_five, case)
knitr::kable(
  uvssamples_five[ds == "MNIST" & pl == "Laplace approximation"], "latex"
)
knitr::kable(
  uvssamples_five[ds == "MNIST" & pl == "deep ensemble"], "latex"
)
knitr::kable(
  uvssamples_five[ds == "CIFAR10" & pl == "Laplace approximation"], "latex"
)
knitr::kable(
  uvssamples_five[ds == "CIFAR10" & pl == "deep ensemble"], "latex"
)

# PLOT UNCERTAINTY VS RESOLUTION -----------------------------------------------

fls = list.files(results_dir, pattern = "uvsresolution.csv", recursive = TRUE)
dt_uvsresolution = lapply(
  fls, function(i) unique(fread(paste(results_dir, i, sep = "/"), drop = 1))
)
dt_uvsresolution = aggregate_dts(dt_uvsresolution)

make_resolution_plot = function(dt, title, col_width, bar_width, ax_factor) {
  make_line_plot(
    dt, 
    "image resolution in % of total",
    title,
    col_width,
    bar_width,
    ax_factor
  ) +
    scale_x_reverse(breaks = unique(dt$case))
}

ggsave(
  paste0(plot_dir, "uvsresolution_mnist_laplace.png"),
  make_resolution_plot(
    dt_uvsresolution[pl == "laplace" & ds == "MNIST"], 
    "MNIST, Laplace approximation",
    2,
    1,
    1
  ),
  height = 2,
  width = 5
)
ggsave(
  paste0(plot_dir, "uvsresolution_mnist_laplace_eu_full.png"),
  make_density_plot(
    as.character(sort(unique(dt_uvsresolution$case))),
    "resolution",
    "laplace",
    "mnist_standard",
    "eu",
    "EU vs image resolution in % (MNIST, Laplace approximation)",
    4
  ),
  height = 3,
  width = 6
)

ggsave(
  paste0(plot_dir, "uvsresolution_mnist_ensemble.png"),
  make_resolution_plot(
    dt_uvsresolution[pl == "ensemble" & ds == "MNIST"], 
    "MNIST, deep ensemble",
    2,
    1,
    1
  ),
  height = 2,
  width = 5
)
ggsave(
  paste0(plot_dir, "uvsresolution_cifar_laplace.png"),
  make_resolution_plot(
    dt_uvsresolution[pl == "laplace" & ds == "CIFAR10"], 
    "CIFAR10, Laplace approximation",
    2,
    1,
    1
  ),
  height = 2,
  width = 5
)
ggsave(
  paste0(plot_dir, "uvsresolution_cifar_ensemble.png"),
  make_resolution_plot(
    dt_uvsresolution[pl == "ensemble" & ds == "CIFAR10"], 
    "CIFAR10, deep ensemble",
    2,
    1,
    1
  ),
  height = 2,
  width = 5
)

uvsresolution_five = lapply(
  seq_len(nrow(configs)),
  function(i) {
    make_tables_five(
      sort(unique(dt_uvsresolution$case)), 
      "resolution", 
      configs[i, 1], 
      configs[i, 2], 
      configs[i, 3], 
      "image resolution"
    )
  }
)
uvsresolution_five = do.call(rbind, uvsresolution_five)
uvsresolution_five$ds = ifelse(
  uvsresolution_five$ds == "mnist_standard", "MNIST", "CIFAR10"
)
uvsresolution_five = uvsresolution_five[
  dt_uvsresolution[
    metric %in% c("tu", "au", "eu"), .(case, pl, ds, metric, mean_value)
  ], 
  on = list(case, pl, ds, metric)
][, metric := toupper(metric)
][, mean_value := round(mean_value, 4)]
uvsresolution_five$pl = ifelse(
  uvsresolution_five$pl == "ensemble", "deep ensemble", "Laplace approximation"
)
uvsresolution_five = unique(uvsresolution_five)
setorder(uvsresolution_five, case)
knitr::kable(
  uvsresolution_five[ds == "MNIST" & pl == "Laplace approximation"], "latex"
)
knitr::kable(
  uvsresolution_five[ds == "MNIST" & pl == "deep ensemble"], "latex"
)
knitr::kable(
  uvsresolution_five[ds == "CIFAR10" & pl == "Laplace approximation"], "latex"
)
knitr::kable(
  uvsresolution_five[ds == "CIFAR10" & pl == "deep ensemble"], "latex"
)

# PLOT SYNTHETIC ---------------------------------------------------------------

fls = list.files(results_dir, pattern = "uvssynthetic_au.csv", recursive = TRUE)
dt_uvssynthetic_au = lapply(
  fls, function(i) unique(fread(paste(results_dir, i, sep = "/"), drop = 1))
)
dt_uvssynthetic_au = aggregate_dts(dt_uvssynthetic_au)

fls = list.files(
  results_dir, pattern = "uvssynthetic_ood.csv", recursive = TRUE
)
dt_uvssynthetic_ood = lapply(
  fls, function(i) unique(fread(paste(results_dir, i, sep = "/"), drop = 1))
)
dt_uvssynthetic_ood = aggregate_dts(dt_uvssynthetic_ood)

make_synth_plot = function(dt, xlab) {
  ggplot(
    dt[metric %in% c("tu", "au", "eu")],
    aes(x = as.factor(case), y = mean_value, fill = metric)
  ) +
    theme_classic() +
    geom_col(position = "dodge", width = 0.4) +
    scale_fill_manual(
      "",
      labels = metric_names,
      values = c("cornflowerblue", "darkorange1", "black")
    ) +
    labs(
      x = xlab, 
      y = "uncertainty"
    )
}

ggsave(
  paste0(plot_dir, "uvssynth_laplace.png"),
  grid.arrange(
    make_line_plot(
      dt_uvssynthetic_au[pl == "laplace"], 
      "side-length ratio",
      "Laplace approximation",
      0.2,
      0.2,
      1
    ) +
      scale_x_continuous(
        breaks = c(1, 2, 5), labels = sprintf("%i:1", c(1, 2, 5))
      ),
    make_synth_plot(dt_uvssynthetic_ood[pl == "laplace"], "") +
      scale_x_discrete(
        labels = c("in-distribution", "out-of-distribution")
      ) +
      theme(
        axis.text.x = element_text(angle = 30, hjust = 1), 
        legend.position = "none"
      ), 
    layout_matrix = matrix(c(1, 1, 2), nrow = 1)
  ),
  height = 1.8,
  width = 6
)
ggsave(
  paste0(plot_dir, "uvssynth_ensemble.png"),
  grid.arrange(
    make_line_plot(
      dt_uvssynthetic_au[pl == "ensemble"], 
      "side-length ratio",
      "Deep ensemble",
      0.2,
      0.2,
      2.5
    ) +
      scale_x_continuous(
        breaks = c(1, 2, 5), labels = sprintf("%i:1", c(1, 2, 5))
      ),
    make_synth_plot(dt_uvssynthetic_ood[pl == "ensemble"], "") +
      scale_x_discrete(
        labels = c("in-distribution", "out-of-distribution")
      ) +
      theme(
        axis.text.x = element_text(angle = 30, hjust = 1), 
        legend.position = "none"
      ), 
    layout_matrix = matrix(c(1, 1, 2), nrow = 1)
  ),
  height = 1.8,
  width = 6
)

configs = expand.grid(
  c("laplace", "ensemble"), c("mnist_standard"), c("tu", "au", "eu")
)
uvssynthetic_au_five = lapply(
  seq_len(nrow(configs)),
  function(i) {
    make_tables_five(
      sort(unique(dt_uvssynthetic_au$case)), 
      "synthetic_au", 
      configs[i, 1], 
      configs[i, 2], 
      configs[i, 3], 
      "class interpolation"
    )
  }
)
uvssynthetic_au_five = do.call(rbind, uvssynthetic_au_five)
uvssynthetic_au_five$ds = ifelse(
  uvssynthetic_au_five$ds == "mnist_standard", "MNIST", "CIFAR10"
)
uvssynthetic_au_five = uvssynthetic_au_five[
  dt_uvssynthetic_au[
    metric %in% c("tu", "au", "eu"), .(case, pl, ds, metric, mean_value)
  ], 
  on = list(case, pl, ds, metric)
][, metric := toupper(metric)
][, mean_value := round(mean_value, 4)]
uvssynthetic_au_five$pl = ifelse(
  uvssynthetic_au_five$pl == "ensemble", "deep ensemble", "Laplace approximation"
)
uvssynthetic_au_five$ds = "synthetic rectangles"
uvssynthetic_au_five = unique(uvssynthetic_au_five)
setorder(uvssynthetic_au_five, case)
knitr::kable(uvssynthetic_au_five[pl == "Laplace approximation"], "latex")
knitr::kable(uvssynthetic_au_five[pl == "deep ensemble"], "latex")


# PLOT RF RESULTS---------------------------------------------------------------

fls_data = list.files(results_dir, pattern = "dataset.csv", recursive = TRUE)
dt_datasets = lapply(
  fls_data, 
  function(i) unique(fread(paste(results_dir, i, sep = "/"), drop = 1))
)
dt_datasets = do.call(rbind, dt_datasets)
dt_datasets$y = as.factor(dt_datasets$y)

fls = list.files(results_dir, pattern = "uvsdistance.csv", recursive = TRUE)
dt_uvsdistance = lapply(
  fls, function(i) unique(fread(paste(results_dir, i, sep = "/"), drop = 1))
)
dt_uvsdistance = aggregate_dts(dt_uvsdistance)
ggsave(
  paste0(plot_dir, "uvsdistance_rf.png"),
  make_line_plot(
    dt_uvsdistance[pl == "rf"], 
    "relative class distance (%)", 
    "Random forest", 
    12,
    12,
    1
  ) + scale_x_reverse(),
  height = 2,
  width = 5
)
ggsave(
  paste0(plot_dir, "uvsdistance_mlp.png"),
  make_line_plot(
    dt_uvsdistance[pl == "mlp"], 
    "relative class distance (%)", 
    "MLP ensemble", 
    12,
    12,
    1
  ) + scale_x_reverse(),
  height = 2,
  width = 5
)
ggsave(
  paste0(plot_dir, "uvsdistance_data.png"),
  make_dataset_plots(
    dt_datasets[rs == 123 & experiment == "distance"], 
    "Data for varying relative class distance", 
    3
  ),
  height = 2.5,
  width = 5
)

fls = list.files(results_dir, pattern = "uvsnoise.csv", recursive = TRUE)
dt_uvsnoise = lapply(
  fls, function(i) unique(fread(paste(results_dir, i, sep = "/"), drop = 1))
)
dt_uvsnoise = aggregate_dts(dt_uvsnoise)
ggsave(
  paste0(plot_dir, "uvsnoise_rf.png"),
  make_line_plot(
    dt_uvsnoise[pl == "rf"], 
    "share of perturbed class labels (%)",
    "Random forest", 
    2,
    2,
    1
  ),
  height = 2,
  width = 5
)
ggsave(
  paste0(plot_dir, "uvsnoise_mlp.png"),
  make_line_plot(
    dt_uvsnoise[pl == "mlp"], 
    "share of perturbed class labels (%)",
    "MLP ensemble", 
    2,
    2,
    1
  ),
  height = 2,
  width = 5
)
ggsave(
  paste0(plot_dir, "uvsnoise_data.png"),
  make_dataset_plots(
    dt_datasets[rs == 123 & experiment == "noise"], 
    "Data for varying noise level", 
    3
  ),
  height = 2.5,
  width = 5
)

fls = list.files(results_dir, pattern = "uvsmembers.csv", recursive = TRUE)
dt_uvsmembers = lapply(
  fls, function(i) unique(fread(paste(results_dir, i, sep = "/"), drop = 1))
)
dt_uvsmembers = aggregate_dts(dt_uvsmembers)
ggsave(
  paste0(plot_dir, "uvsmembers_rf.png"),
  make_line_plot(
    dt_uvsmembers[pl == "rf"], 
    "number of ensemble members",
    "Random forest", 
    1.5,
    1.5,
    1
  ),
  height = 2,
  width = 5
)
ggsave(
  paste0(plot_dir, "uvsmembers_mlp.png"),
  make_line_plot(
    dt_uvsmembers[pl == "mlp"], 
    "number of ensemble members",
    "MLP ensemble", 
    1.5,
    1.5,
    1
  ),
  height = 2,
  width = 5
)

fls = list.files(results_dir, pattern = "uvscomplexity.csv", recursive = TRUE)
dt_uvscomplexity = lapply(
  fls, function(i) unique(fread(paste(results_dir, i, sep = "/"), drop = 1))
)
dt_uvscomplexity = aggregate_dts(dt_uvscomplexity)
ggsave(
  paste0(plot_dir, "uvscomplexity_rf.png"),
  make_line_plot(
    dt_uvscomplexity[pl == "rf"], 
    "maximum tree depth",
    "Random forest", 
    0.5,
    0.5,
    1
  ),
  height = 2,
  width = 5
)
ggsave(
  paste0(plot_dir, "uvscomplexity_mlp.png"),
  make_line_plot(
    dt_uvscomplexity[pl == "mlp"], 
    "number of hidden-layer neurons",
    "MLP ensemble", 
    0.5,
    0.5,
    1
  ),
  height = 2,
  width = 5
)

# PLOT TOY Q DISTRIBUTIONS -----------------------------------------------------

n_samples = 100000

make_title = function(x) {
  sprintf(
    "TU: %.2f, AU: %.2f, EU: %.2f",
    metrics[distro == x, tu],
    metrics[distro == x, au],
    metrics[distro == x, eu]
  )
}

set.seed(123)
thetas_norm = rnorm(n_samples, mean = 0.5, sd = 0.1)
thetas_norm = thetas_norm[thetas_norm >= 0 & thetas_norm <= 1]
thetas_beta = rbeta(n_samples, 8, 2)
thetas_beta = thetas_beta[thetas_beta >= 0 & thetas_beta <= 1]
thetas_unif_1 = runif(n_samples)
thetas_unif_2 = runif(n_samples, 0.3, 0.7)
thetas_unif_3 = runif(n_samples, 0.45, 0.85)

metrics = data.table(
  distro = c("norm", "beta", "unif_1", "unif_2", "unif_3", "dirac"),
  tu = c(1, 0.8, 1, 1, compute_binary_entropy(0.6), 1),
  au = c(
    mean(sapply(thetas_norm, compute_binary_entropy)),
    mean(sapply(thetas_beta, compute_binary_entropy)),
    mean(sapply(thetas_unif_1, compute_binary_entropy)),
    mean(sapply(thetas_unif_2, compute_binary_entropy)),
    mean(sapply(thetas_unif_3, compute_binary_entropy)),
    0
  )
)
metrics[, eu := tu - au]

base_plot = ggplot() +
  theme_classic() +
  labs(x = expression(theta), y = "") +
  scale_x_continuous(breaks = seq(0, 1, by = 0.2)) +
  scale_y_continuous(breaks = NULL) +
  theme(plot.title = element_text(hjust = 0.5))

my_col = "darkgray"
  
p_norm = base_plot +
  stat_function(
    fun = dnorm, args = list(mean = 0.5, sd = 0.1), col = my_col
  ) +
  ggtitle(make_title("norm"))
p_beta = base_plot +
  stat_function(
    fun = dbeta, args = list(shape1 = 8, shape2 = 2), col = my_col
  ) +
  ggtitle(make_title("beta"))
p_unif_1 = base_plot +
  stat_function(fun = dunif, col = my_col) +
  ggtitle(make_title("unif_1"))
p_unif_2 = base_plot +
  stat_function(
    fun = dunif, args = list(min = 0.3, max = 0.7), col = my_col
  ) +
  ggtitle(make_title("unif_2"))
p_unif_3 = base_plot +
  stat_function(
    fun = dunif, args = list(min = 0.45, max = 0.85), col = my_col
  ) +
  ggtitle(make_title("unif_3"))
p_dirac = ggplot(data.frame(x = 0:1, y = 1), aes(x, ymax = y, ymin = 0)) +
  geom_linerange(col = my_col) +
  theme_classic() +
  labs(x = expression(theta), y = "") +
  scale_x_continuous(breaks = seq(0, 1, by = 0.2)) +
  scale_y_continuous(breaks = NULL) +
  theme(plot.title = element_text(hjust = 0.5)) +
  ggtitle(make_title("dirac"))

p = grid.arrange(
  p_unif_1, p_norm, p_beta, p_unif_2, p_unif_3, p_dirac, ncol = 2
)

ggsave(paste0(plot_dir, "q_distributions.png"), p, height = 4, width = 5.5)

# PLOT LEARNING EXAMPLE --------------------------------------------------------

n_samples = 2000
set.seed(pi)
class_1 = rmvnorm(n_samples, c(1, 2), sigma = diag(c(0.1, 0.2)))
class_2 = rmvnorm(n_samples, c(2, 1), sigma = diag(c(0.1, 0.1)))
dt = data.table(
  x_1 = c(class_1[, 1], class_2[, 1]),
  x_2 = c(class_1[, 2], class_2[, 2]),
  class = c(rep(0, n_samples), rep(1, n_samples)),
  shape = 1
)
dt = rbind(dt, t(c(x_1 = 1.4, x_2 = 2, class = 2, shape = 2)))

make_point_cloud = function(dt, alphas = c(0.5, 1)) {
  ggplot(
    dt, aes(x_1, x_2, col = as.factor(class), alpha = as.factor(shape)),
  ) +
    geom_point(size = 1.1) +
    theme_classic() +
    labs(x = "", y = "") +
    scale_x_continuous(breaks = NULL) +
    scale_y_continuous(breaks = NULL) +
    theme(axis.line = element_blank(), legend.position = "none") +
    scale_color_manual(values = c("#440154FF", "#2A788EFF", "#7AD151FF")) +
    scale_alpha_manual(values = alphas)
}

ggsave(
  paste0(plot_dir, "point_cloud_0.png"), 
  make_point_cloud(dt[c(1, 3000, 4001)], c(1, 1)),
  height = 1,
  width = 1.5
)
ggsave(
  paste0(plot_dir, "point_cloud_1.png"), 
  make_point_cloud(dt),
  height = 2,
  width = 3
)