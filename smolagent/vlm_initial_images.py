from io import BytesIO
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image
from smolagents import CodeAgent, OpenAIServerModel

LOCAL_IMAGE_DIR = Path("images")
LOCAL_IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.webp")
IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg",
    "https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg",
]


def load_local_images(image_dir: Path) -> list[Image.Image]:
    image_paths: list[Path] = []
    for pattern in LOCAL_IMAGE_PATTERNS:
        image_paths.extend(sorted(image_dir.glob(pattern)))

    images: list[Image.Image] = []
    for image_path in image_paths:
        images.append(Image.open(image_path).convert("RGB"))
    return images


def load_images(urls: list[str]) -> list[Image.Image]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        )
    }
    images: list[Image.Image] = []
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            raise RuntimeError(
                "Image download failed. Put guest images into the local "
                f"'{LOCAL_IMAGE_DIR}' folder to avoid remote rate limits. "
                f"Failed URL: {url}. HTTP status: {status}."
            ) from exc
        images.append(Image.open(BytesIO(response.content)).convert("RGB"))
    return images


def main() -> None:
    load_dotenv()

    images = load_local_images(LOCAL_IMAGE_DIR)
    if images:
        print(f"Loaded {len(images)} local images from {LOCAL_IMAGE_DIR}.")
    else:
        print(f"No local images found in {LOCAL_IMAGE_DIR}; downloading sample images.")
        images = load_images(IMAGE_URLS)

    model = OpenAIServerModel(model_id="gpt-4o")
    agent = CodeAgent(
        tools=[],
        model=model,
        max_steps=20,
        verbosity_level=2,
    )
    response = agent.run(
        """
        Describe the costume and makeup that the comic character in these
        photos is wearing and return the description.
        Tell me if the guest is The Joker or Wonder Woman.
        """,
        images=images,
    )
    print(response)


if __name__ == "__main__":
    main()
