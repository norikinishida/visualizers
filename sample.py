import numpy as np

import visualizers

# Plot
legend_names = ["sin", "cos"]
xs = np.linspace(-np.pi, np.pi)
sin_ys = np.sin(xs)
cos_ys = np.cos(xs)
list_ys = [sin_ys, cos_ys]
list_xs = [xs, xs]
visualizers.plot(
    list_ys=list_ys, list_xs=list_xs,
    xticks=None, xlabel="xlabel", ylabel="ylabel",
    legend_names=legend_names, legend_anchor=(1.0, 1.0), legend_location="upper right",
    marker="o", linestyle="-", markersize=10,
    fontsize=30,
    savepath=None, figsize=(8,6), dpi=100)

# Plot with error bar
legend_names = ["sin", "cos"]
xs = np.linspace(-np.pi, np.pi)
sin_ys = np.sin(xs)
cos_ys = np.cos(xs)
sin_es = np.random.normal(0.0, 0.1, sin_ys.shape)
cos_es = np.random.normal(0.0, 0.1, cos_ys.shape)
list_ys = [sin_ys, cos_ys]
list_es = [sin_es, cos_es]
list_xs = [xs, xs]
visualizers.errorbar(
    list_ys=list_ys, list_es=list_es, list_xs=list_xs,
    xticks=None, xlabel="xlabel", ylabel="ylabel",
    legend_names=legend_names, legend_anchor=(1.0, 1.0), legend_location="upper right",
    marker="o", linestyle="-", markersize=10,
    capsize=4.0, capthick=2.0,
    fontsize=30,
    savepath=None, figsize=(8,6), dpi=100)

# Scatter (without/with mean positions and covariance regions)
mean1 = np.asarray([1.0, 1.0])
mean2 = np.asarray([-1.0, -1.0])
cov = np.asarray([[1.0, 0.0],[0.0,1.0]])
vectors1 = np.random.multivariate_normal(mean1, cov, 200)
vectors2 = np.random.multivariate_normal(mean2, cov, 200)
vectors = np.vstack([vectors1, vectors2])
categories = np.asarray(["A" for _ in range(200)] + ["B" for _ in range(200)])
category_name = "Category"
category_order = ["A", "B"]
category_centers = [mean1, mean2]
category_covariances = [cov, cov]
visualizers.scatter(
    vectors=vectors,
    categories=categories, category_name=category_name, category_order=category_order,
    category_centers=None, category_covariances=None,
    xlabel="xlabel", ylabel="ylabel",
    fontsize=30,
    savepath=None, figsize=(8,6), dpi=100)
visualizers.scatter(
    vectors=vectors,
    categories=categories, category_name=category_name, category_order=category_order,
    category_centers=category_centers, category_covariances=category_covariances,
    xlabel="xlabel", ylabel="ylabel",
    fontsize=30,
    savepath=None, figsize=(8,6), dpi=100)

# Bar
legend_names = ["model1", "model2", "model3"]
xticks = ["metric1", "metric2", "metric3", "metric4", "metric5"]
list_ys = []
for _ in range(len(legend_names)):
    ys = np.random.random((len(xticks),)).tolist()
    list_ys.append(ys)
visualizers.bar(
    list_ys=list_ys,
    xticks=xticks, xlabel="xlabel", ylabel="ylabel",
    legend_names=legend_names, legend_anchor=(1.0, 1.0), legend_location="upper right",
    fontsize=30,
    savepath=None, figsize=(8,6), dpi=100)

# Heatmap
matrix = np.random.random((6,5))
xticks = ["x1", "x2", "x3", "x4", "x5"]
yticks = ["y1", "y2", "y3", "y4", "y5", "y6"]
visualizers.heatmap(
    matrix=matrix,
    xticks=xticks, yticks=yticks, xlabel="xlabel", ylabel="ylabel",
    vmin=None, vmax=None,
    annotate_counts=True, show_colorbar=True, colormap="Blues",
    linewidths=0, fmt=".2g",
    fontsize=30,
    savepath=None, figsize=(8,6), dpi=100)

