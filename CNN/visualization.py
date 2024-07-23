import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image


def plot_class_distribution(idm_df_filtered):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='BINARY STATUS', data=idm_df_filtered)
    plt.title('Class Distribution')
    for p in ax.patches:
        count = int(p.get_height())
        ax.annotate(f'{count}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.show()


def show_sample_images(day_image_data, night_image_data, class_label, n=5, cols=2):
    class_images_day = day_image_data[day_image_data['status'] == class_label]
    class_images_night = night_image_data[night_image_data['status'] == class_label]

    common_villages = pd.merge(class_images_day, class_images_night, on='id')
    sample_villages = common_villages.sample(n)

    rows = (n // cols) + (n % cols > 0)
    fig, axes = plt.subplots(rows, cols * 2, figsize=(30, 5 * rows))
    fig.patch.set_facecolor('black')
    axes = axes.flatten()

    for i, (_, village_row) in enumerate(sample_villages.iterrows()):
        # Day images
        img_path_day = village_row['filepath_x']
        img_day = Image.open(img_path_day)
        ax_day = axes[i * 2]
        ax_day.imshow(img_day)
        ax_day.set_title(f"Day - {class_label} - {village_row['kecamatan_x']}_{village_row['desa_x']}", color='white')
        ax_day.axis('off')
        ax_day.set_facecolor('black')

        # Night images
        img_path_night = village_row['filepath_y']
        img_night = Image.open(img_path_night)
        ax_night = axes[i * 2 + 1]
        ax_night.imshow(img_night)
        ax_night.set_title(f"Night - {class_label} - {village_row['kecamatan_y']}_{village_row['desa_y']}",
                           color='white')
        ax_night.axis('off')
        ax_night.set_facecolor('black')

    for ax in axes[len(sample_villages) * 2:]:
        ax.axis('off')
        ax.set_facecolor('black')

    plt.tight_layout()
    plt.show()
