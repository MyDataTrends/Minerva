from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image, ImageDraw

def decode_secret_message(url, driver_path):
    options = Options()
    options.use_chromium = True
    service = Service(driver_path)
    driver = webdriver.Edge(service=service, options=options)
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        table = driver.find_element(By.TAG_NAME, "table")
        rows = table.find_elements(By.TAG_NAME, "tr")
        coordinates = []
        for row in rows[1:]:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) == 3:
                x = int(cells[0].text)
                unicode_character = cells[1].text
                y = int(cells[2].text)
                coordinates.append((x, y, unicode_character))
        max_x = max(coord[0] for coord in coordinates)
        max_y = max(coord[1] for coord in coordinates)
        grid_size = 10
        img_width = (max_x + 1) * grid_size
        img_height = (max_y + 1) * grid_size
        img = Image.new("RGB", (img_width, img_height), "white")
        draw = ImageDraw.Draw(img)
        for x, y, char in coordinates:
            dot_x = x * grid_size
            dot_y = y * grid_size
            draw.text((dot_x, dot_y), char, fill="black")
        output_file = "decoded_message.jpg"
        img.save(output_file)
        print(f"Decoded message saved to {output_file}")
    finally:
        driver.quit()


def main():
    # Example usage - update driver_path accordingly
    decode_secret_message(
        "https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub",
        driver_path=r"C:\\Path\\To\\msedgedriver.exe",
    )


if __name__ == "__main__":
    main()
