# 📄 Twitter Hashtag Extraction

This project contains Python scripts designed for extracting specific hashtags from Twitter. The scripts leverage web scraping and API integration to identify and gather data related to targeted hashtags.

---

## 📂 Project Overview
The repository includes the following Python files:

1. **TwitterMiner.py**: A script that uses Twitter API to fetch and filter tweets containing specific hashtags.
2. **twitterWebScrapper.py**: A web scraper that extracts tweets with specific hashtags from Twitter’s web interface.

These tools can be used individually or combined to gather hashtag data for analysis, research, or monitoring trends.

---

## 🛠️ Prerequisites

Ensure the following dependencies are installed before running the scripts:

- Python 3.6+
- Tweepy (if using Twitter API)
- BeautifulSoup4 (if web scraping)
- Requests
- Selenium (for web scraping with dynamic content)

Install the dependencies using:
```bash
pip install tweepy beautifulsoup4 requests selenium
```

---

## 🚀 How to Use

### 1. **TwitterMiner.py**
This script uses the Twitter API to collect tweets with specific hashtags. Ensure you have a valid Twitter Developer Account and API credentials set up:

- **API Key**
- **API Secret Key**
- **Access Token**
- **Access Token Secret**

#### Steps:
1. Replace the placeholder credentials in the script with your own.
2. Run the script:
   ```bash
   python TwitterMiner.py
   ```
3. Specify the hashtags and other parameters as needed.

### 2. **twitterWebScrapper.py**
This script uses web scraping to gather tweets containing specific hashtags from Twitter’s public web pages.

#### Steps:
1. Ensure Selenium WebDriver (e.g., ChromeDriver) is installed and set up.
2. Run the script:
   ```bash
   python twitterWebScrapper.py
   ```
3. Modify the script to specify the hashtags and search parameters.

---

## ⚠️ Legal and Ethical Considerations
Ensure that your use of these scripts complies with Twitter’s terms of service and data privacy regulations. Excessive scraping or misuse of API resources can lead to your account being suspended.

---

## 📧 Contact
For questions or suggestions, feel free to reach out:
- 📧 mfazli@stanford.edu
- 📧 mfazli@meei.harvard.edu

Happy coding and hashtag hunting! 🚀
