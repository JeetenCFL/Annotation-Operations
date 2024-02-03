import json
import pandas as pd
import argparse


def merge_coco_annotations(annotation_files):
    """Merges COCO annotation data from multiple JSON files into a single JSON file.

    Args:
        annotation_files (list): A list of paths to COCO annotation JSON files.

    Returns:
        None
    """

    faulty_images = []

    # Load and merge data into a single DataFrame
    df = pd.DataFrame()
    for file_path in annotation_files:
        with open(file_path, "r") as f:
            anno = json.load(f)

        temp_df = pd.DataFrame(
            columns=[
                "imgID",
                "imgPath",
                "dimensions",
                "areas",
                "labels",
                "polygons",
                "bboxes",
            ]
        )
        for i in range(len(anno["images"])):
            temp_dict = {
                "imgID": anno["images"][i]["id"],
                "imgPath": anno["images"][i]["file_name"],
                "dimensions": (anno["images"][i]["height"], anno["images"][i]["width"]),
                "areas": [],
                "labels": [],
                "polygons": [],
                "bboxes": [],
            }
            # Use concat() to add rows to the DataFrame:
            temp_df = pd.concat([temp_df, pd.DataFrame([temp_dict])], ignore_index=True)

        temp_df.set_index("imgID", inplace=True)

        for i in range(len(anno["annotations"])):
            loc_index = int(anno["annotations"][i]["image_id"])
            try:
                temp_df.loc[loc_index]["areas"].append(anno["annotations"][i]["area"])
                temp_df.loc[loc_index]["labels"].append(
                    anno["annotations"][i]["category_id"]
                )
                temp_df.loc[loc_index]["polygons"].append(
                    anno["annotations"][i]["segmentation"]
                )
                temp_df.loc[loc_index]["bboxes"].append(anno["annotations"][i]["bbox"])
            except:
                faulty_images.append(loc_index)

        temp_df.sort_index(inplace=True)

        # Use concat() to combine DataFrames:
        df = pd.concat([df, temp_df], ignore_index=True)

    # Remove images without bounding boxes
    df = df[df["bboxes"].apply(len) > 0]

    # Create combined COCO JSON
    combined_json = {
        "annotations": [],
        "categories": anno[
            "categories"
        ],  # Assuming categories are consistent across files
        "images": [],
        "info": {},
        "licenses": [],
    }
    anno_id_count = 0
    for img_id, row in df.iterrows():
        temp_images = {
            "file_name": row["imgPath"],
            "height": row["dimensions"][0],
            "width": row["dimensions"][1],
            "id": img_id,
        }
        combined_json["images"].append(temp_images)
        for i in range(len(row["labels"])):
            temp_anno = {
                "area": row["areas"][i],
                "bbox": row["bboxes"][i],
                "category_id": row["labels"][i],
                "id": anno_id_count,
                "image_id": img_id,
                "iscrowd": 0,
                "segmentation": [row["polygons"][i]],
            }
            combined_json["annotations"].append(temp_anno)
            anno_id_count += 1

    # Save combined JSON
    with open("CombinedCoco2.json", "w") as outfile:
        json.dump(combined_json, outfile)

    # Print paths of faulty images
    for img_id in faulty_images:
        print(df.loc[img_id]["imgPath"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge COCO annotation files.")
    parser.add_argument(
        "annotation_files", nargs="+", help="Paths to COCO annotation JSON files."
    )
    args = parser.parse_args()

    merge_coco_annotations(args.annotation_files)