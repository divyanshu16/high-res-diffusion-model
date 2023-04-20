from datasets import load_dataset


if __name__=="__main__":
    dataset = load_dataset("laion/laion-high-resolution", cache_dir='.')
    dataset.save_to_disk('laon_downloaded')
