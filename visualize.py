import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_points(colors):
    model.eval()
    z = model(torch.arange(data.num_nodes))
    z = TSNE(n_components=2).fit_transform(z.detach().numpy())
    y = data.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(4):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    plt.axis('off')
    plt.show()

colors = [
    '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
]
# plot_points(colors)