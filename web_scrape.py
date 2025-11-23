import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Setup Chrome driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.maximize_window()
driver.get("https://www.myscheme.gov.in/search")
time.sleep(5)

scheme_data = []
page_count = 1

def scrape_page(soup):
    schemes = soup.select('div.group')
    for scheme in schemes:
        try:
            name_tag = scheme.select_one('h2 a span')
            name = name_tag.text.strip() if name_tag else ""

            state_tag = scheme.select('h2')
            state = state_tag[1].text.strip() if len(state_tag) > 1 else ""

            desc_tag = scheme.select_one('span.line-clamp-2 span')
            description = desc_tag.text.strip() if desc_tag else ""

            tag_elements = scheme.select('div[title] span')
            tags = [tag.text.strip() for tag in tag_elements]

            scheme_data.append({
                'name': name,
                'state': state,
                'description': description,
                'tags': tags
            })
        except Exception as e:
            print(f"Skipping a scheme due to error: {e}")

# Loop: click the Next button until it disappears or fails
while True:
    try:
        # Wait for schemes to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.group"))
        )
        time.sleep(1)

        print(f"Scraping page {page_count}")
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        scrape_page(soup)
        page_count += 1

        # Wait for Next button to appear using CSS selector (no XPath)
        next_svg = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((
                By.CSS_SELECTOR, 'ul.list-none svg.cursor-pointer:last-of-type'
            ))
        )

        # Scroll into view to prevent click interception
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_svg)
        time.sleep(0.5)

        # Try normal click, fallback to JS click
        try:
            next_svg.click()
        except:
            driver.execute_script("arguments[0].click();", next_svg)

        time.sleep(2)  # Wait for next page to load

    except Exception as e:
        print(f"Finished scraping or next button failed: {e}")
        break

# Done scraping
driver.quit()

# Save data
df = pd.DataFrame(scheme_data)
df.to_csv('myscheme_data.csv', index=False)

print(f"Scraping complete. Pages scraped: {page_count - 1}, Total schemes: {len(scheme_data)}")
