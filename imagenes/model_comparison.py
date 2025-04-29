import matplotlib.pyplot as plt
import numpy as np


def main():
    # Datos de rendimiento de modelos (basados en el paper)
    models = ["ResNet50", "EfficientNetB3", "MobileNetV2", "VGG16", "InceptionV3"]
    accuracy = [92.8, 91.5, 89.3, 87.2, 90.1]  # Precisión (%)
    inference_time = [45, 25, 15, 65, 47]  # Tiempo de inferencia (ms)
    parameters = [25.6, 12.3, 3.5, 138.4, 23.9]  # Millones de parámetros

    # Tamaño de los puntos proporcional al número de parámetros
    sizes = [p * 5 for p in parameters]

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 8))

    # Crear scatter plot
    scatter = ax.scatter(
        inference_time,
        accuracy,
        s=sizes,
        alpha=0.6,
        c=range(len(models)),
        cmap="viridis",
        edgecolors="black",
    )

    # Añadir etiquetas para cada punto
    for i, model in enumerate(models):
        ax.annotate(
            model,
            (inference_time[i], accuracy[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    # Añadir líneas de referencia
    ax.axhline(y=90, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=30, color="gray", linestyle="--", alpha=0.3)

    # Dividir el gráfico en cuadrantes con anotaciones
    ax.text(
        10,
        93,
        "IDEAL\n(Rápido y Preciso)",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="lightgreen", alpha=0.1, boxstyle="round"),
    )
    ax.text(
        50,
        93,
        "PRECISO\n(Lento pero Preciso)",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="lightyellow", alpha=0.1, boxstyle="round"),
    )
    ax.text(
        10,
        86,
        "RÁPIDO\n(Rápido pero menos Preciso)",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="lightyellow", alpha=0.1, boxstyle="round"),
    )
    ax.text(
        50,
        86,
        "SUBÓPTIMO\n(Lento y menos Preciso)",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="lightcoral", alpha=0.1, boxstyle="round"),
    )

    # Configurar ejes
    ax.set_xlabel("Tiempo de inferencia promedio (ms)", fontsize=12)
    ax.set_ylabel("Precisión en test (%)", fontsize=12)
    ax.set_title(
        "Comparación de Modelos CNN para Clasificación de Vehículos",
        fontsize=14,
        fontweight="bold",
    )

    # Configurar límites de los ejes
    ax.set_xlim(0, 80)
    ax.set_ylim(85, 95)

    # Leyenda para el tamaño de los puntos
    sizes_legend = [5, 25, 50, 100]
    labels = ["1M", "5M", "10M", "20M"]

    # Crear leyenda para el tamaño de los puntos (parámetros)
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"{label} parámetros",
            markerfacecolor="gray",
            markersize=np.sqrt(size) / 2,
        )
        for size, label in zip(sizes_legend, labels)
    ]

    ax.legend(
        handles=legend_elements,
        title="Tamaño del modelo",
        loc="lower right",
        title_fontsize=10,
    )

    # Añadir cuadrícula
    ax.grid(True, alpha=0.3)

    # Añadir información adicional
    plt.figtext(
        0.5,
        0.01,
        "Fuente: Experimentos realizados en dataset personalizado de vehículos con 10 categorías. Hardware: NVIDIA RTX 3080.",
        ha="center",
        fontsize=8,
        style="italic",
    )

    # Ajustar layout y guardar
    plt.tight_layout()
    plt.savefig("model_comparison.jpg", dpi=300, bbox_inches="tight")
    print("Model comparison chart saved to 'model_comparison.jpg'")


if __name__ == "__main__":
    main()
