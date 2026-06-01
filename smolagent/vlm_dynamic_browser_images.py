from io import BytesIO
from time import sleep
import helium
from dotenv import load_dotenv
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from smolagents import (
    ActionStep,
    CodeAgent,
    DuckDuckGoSearchTool,
    OpenAIServerModel,
    tool,
)

HELIUM_INSTRUCTIONS = """
You can use helium to browse the web. Use these patterns when needed:
- import helium
- helium.start_chrome("https://example.com")
- helium.go_to("https://example.com")
- helium.click("text")
- helium.write("query")
- helium.press(helium.ENTER)
Rely on screenshots after each step to inspect the current page visually.
"""


@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page and jumps to the nth occurrence.
    Args:
        text: The text to search for.
        nth_result: Which occurrence to jump to.
    """
    driver = helium.get_driver()
    if driver is None:
        raise RuntimeError("Browser is not started. Start Chrome with helium first.")
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if nth_result > len(elements):
        raise ValueError(
            f"Match number {nth_result} not found; only {len(elements)} matches found."
        )
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    return (
        f"Found {len(elements)} matches for '{text}'. Focused on result {nth_result}."
    )


@tool
def go_back() -> str:
    """Goes back to the previous browser page."""
    driver = helium.get_driver()
    if driver is None:
        raise RuntimeError("Browser is not started. Start Chrome with helium first.")
    driver.back()
    return "Went back to the previous page."


@tool
def close_popups() -> str:
    """
    Closes visible modal or pop-up windows with Escape.
    This does not handle every cookie consent banner.
    """
    driver = helium.get_driver()
    if driver is None:
        raise RuntimeError("Browser is not started. Start Chrome with helium first.")
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
    return "Sent Escape to close visible pop-ups."


def save_screenshot(step_log: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)
    driver = helium.get_driver()
    if driver is None:
        return
    current_step = step_log.step_number
    for previous_step in agent.logs:
        if (
            isinstance(previous_step, ActionStep)
            and previous_step.step_number <= current_step - 2
        ):
            previous_step.observations_images = None
    png_bytes = driver.get_screenshot_as_png()
    image = Image.open(BytesIO(png_bytes)).convert("RGB")
    print(f"Captured a browser screenshot: {image.size} pixels")
    step_log.observations_images = [image.copy()]
    url_info = f"Current url: {driver.current_url}"
    if step_log.observations:
        step_log.observations = f"{step_log.observations}\n{url_info}"
    else:
        step_log.observations = url_info


def main() -> None:
    load_dotenv()
    model = OpenAIServerModel(model_id="gpt-4o")
    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool(), go_back, close_popups, search_item_ctrl_f],
        model=model,
        additional_authorized_imports=["helium"],
        step_callbacks=[save_screenshot],
        max_steps=20,
        verbosity_level=2,
    )
    try:
        response = agent.run("""
            I am Alfred, the butler of Wayne Manor, responsible for verifying
            the identity of guests at a party. A superhero has arrived at the
            entrance claiming to be Wonder Woman, but I need to confirm if she
            is who she says she is.
            Please search for images of Wonder Woman and generate a detailed
            visual description based on those images. Additionally, navigate to
            Wikipedia to gather key details about her appearance. With this
            information, I can determine whether to grant her access to the
            event.
            """ + HELIUM_INSTRUCTIONS)
        print(response)
    finally:
        driver = helium.get_driver()
        if driver is not None:
            driver.quit()


if __name__ == "__main__":
    main()
