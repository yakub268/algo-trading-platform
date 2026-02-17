"""
Box Office Data Scraper

Fetches box office data for Kalshi movie contracts:
- Box Office Mojo weekend estimates
- The Numbers tracking data
- Rotten Tomatoes scores for sentiment
- Opening weekend predictions

Author: Trading Bot
Created: January 2026
"""

import os
import json
import logging
import requests
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BoxOfficeScraper')


@dataclass
class MovieData:
    """Box office movie data"""
    title: str
    release_date: Optional[str]
    distributor: Optional[str]
    budget: Optional[float]  # In millions
    opening_weekend: Optional[float]  # Actual or estimate in millions
    total_gross: Optional[float]  # In millions
    theaters: Optional[int]
    rt_score: Optional[int]  # Rotten Tomatoes critic score
    rt_audience: Optional[int]  # Rotten Tomatoes audience score
    metacritic: Optional[int]
    genre: Optional[str]
    is_sequel: bool
    franchise: Optional[str]
    source: str
    fetched_at: datetime


@dataclass
class BoxOfficeProbability:
    """Probability estimate for box office outcome"""
    movie_title: str
    ticker_pattern: str
    threshold: float  # In millions
    direction: str  # 'above' or 'below'
    our_probability: float
    reasoning: str
    confidence: str


# Historical opening weekend data for modeling (in millions)
FRANCHISE_MULTIPLIERS = {
    'Marvel': 1.4,
    'Star Wars': 1.3,
    'DC': 1.2,
    'Disney Animation': 1.2,
    'Pixar': 1.15,
    'Fast & Furious': 1.1,
    'Jurassic': 1.1,
    'Avatar': 1.5,
    'default': 1.0
}

# RT score to box office correlation
RT_SCORE_BOOST = {
    (90, 100): 1.15,
    (80, 90): 1.08,
    (70, 80): 1.0,
    (60, 70): 0.95,
    (50, 60): 0.85,
    (0, 50): 0.75,
}


class BoxOfficeScraper:
    """
    Scrapes box office data from public sources.

    Sources:
    - Box Office Mojo (via web scraping)
    - The Numbers (via web scraping)
    - OMDb API for Rotten Tomatoes scores (free tier)
    """

    BOM_BASE = "https://www.boxofficemojo.com"
    NUMBERS_BASE = "https://www.the-numbers.com"
    OMDB_API = "https://www.omdbapi.com"

    CACHE_DURATION = timedelta(hours=4)

    def __init__(self, omdb_api_key: str = None, cache_dir: str = "data/boxoffice_cache"):
        """Initialize the box office scraper"""
        self.cache_dir = cache_dir
        self.omdb_api_key = omdb_api_key or os.getenv('OMDB_API_KEY')
        os.makedirs(cache_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })

        logger.info("BoxOfficeScraper initialized")

    def _get_cache_path(self, key: str) -> str:
        """Get cache file path"""
        safe_key = re.sub(r'[^\w\-]', '_', key)
        return os.path.join(self.cache_dir, f"{safe_key}.json")

    def _load_cache(self, key: str) -> Optional[Dict]:
        """Load from cache if valid"""
        cache_path = self._get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                cached_time = datetime.fromisoformat(data.get('cached_at', ''))
                if datetime.now(timezone.utc) - cached_time < self.CACHE_DURATION:
                    return data.get('data')
            except Exception as e:
                logger.debug(f"Cache load error for {key}: {e}")
        return None

    def _save_cache(self, key: str, data: any):
        """Save to cache"""
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'cached_at': datetime.now(timezone.utc).isoformat(),
                    'data': data
                }, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"Cache save error for {key}: {e}")

    def _parse_money(self, text: str) -> Optional[float]:
        """Parse money string to float (in millions)"""
        if not text:
            return None
        try:
            # Remove currency symbols and commas
            cleaned = re.sub(r'[$,]', '', text.strip())
            value = float(cleaned)
            # Convert to millions if it's a large number
            if value > 1000:
                return value / 1_000_000
            return value
        except Exception as e:
            logger.debug(f"Error parsing money string '{text}': {e}")
            return None

    def fetch_bom_weekend(self) -> List[Dict]:
        """
        Fetch current weekend box office from Box Office Mojo.

        Returns:
            List of movies with weekend performance
        """
        cache_key = "bom_weekend"
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        try:
            url = f"{self.BOM_BASE}/weekend/chart/"
            response = self.session.get(url, timeout=15)

            if response.status_code != 200:
                logger.warning(f"BOM returned {response.status_code}")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            movies = []

            # Find the main table
            table = soup.find('table', {'class': 'mojo-body-table'})
            if not table:
                table = soup.find('table')

            if table:
                rows = table.find_all('tr')[1:]  # Skip header

                for row in rows[:20]:  # Top 20 movies
                    cells = row.find_all('td')
                    if len(cells) >= 5:
                        title_cell = cells[1] if len(cells) > 1 else cells[0]
                        title = title_cell.get_text(strip=True)

                        movie = {
                            'title': title,
                            'weekend_gross': self._parse_money(cells[2].get_text()) if len(cells) > 2 else None,
                            'theaters': int(re.sub(r'[,]', '', cells[4].get_text())) if len(cells) > 4 else None,
                            'total_gross': self._parse_money(cells[6].get_text()) if len(cells) > 6 else None,
                        }
                        movies.append(movie)

            self._save_cache(cache_key, movies)
            logger.info(f"[BOM] Fetched {len(movies)} movies from weekend chart")
            return movies

        except Exception as e:
            logger.error(f"BOM scraping error: {e}")
            return []

    def fetch_upcoming_releases(self) -> List[Dict]:
        """
        Fetch upcoming movie releases.

        Returns:
            List of upcoming movies with release dates
        """
        cache_key = "bom_upcoming"
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        try:
            # Get upcoming releases from Box Office Mojo
            url = f"{self.BOM_BASE}/release/upcoming/"
            response = self.session.get(url, timeout=15)

            movies = []

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find release entries
                releases = soup.find_all('div', class_='a-section')

                for release in releases[:30]:
                    title_elem = release.find('a', class_='a-link-normal')
                    date_elem = release.find('span', class_='a-size-base')

                    if title_elem:
                        movies.append({
                            'title': title_elem.get_text(strip=True),
                            'release_date': date_elem.get_text(strip=True) if date_elem else None,
                            'source': 'BOM'
                        })

            # Fallback: use hardcoded upcoming releases if scraping fails
            if not movies:
                movies = self._get_known_upcoming()

            self._save_cache(cache_key, movies)
            logger.info(f"[BOM] Found {len(movies)} upcoming releases")
            return movies

        except Exception as e:
            logger.error(f"BOM upcoming error: {e}")
            return self._get_known_upcoming()

    def _get_known_upcoming(self) -> List[Dict]:
        """Get known upcoming releases as fallback"""
        # Major upcoming releases (would be updated periodically)
        return [
            {'title': 'Captain America: Brave New World', 'release_date': '2025-02-14', 'franchise': 'Marvel'},
            {'title': 'Snow White', 'release_date': '2025-03-21', 'franchise': 'Disney'},
            {'title': 'Thunderbolts', 'release_date': '2025-05-02', 'franchise': 'Marvel'},
            {'title': 'Mission: Impossible 8', 'release_date': '2025-05-23', 'franchise': 'Mission Impossible'},
            {'title': 'Jurassic World Rebirth', 'release_date': '2025-07-02', 'franchise': 'Jurassic'},
            {'title': 'Superman', 'release_date': '2025-07-11', 'franchise': 'DC'},
            {'title': 'Fantastic Four', 'release_date': '2025-07-25', 'franchise': 'Marvel'},
            {'title': 'Avatar 3', 'release_date': '2025-12-19', 'franchise': 'Avatar'},
        ]

    def fetch_omdb_data(self, title: str) -> Optional[Dict]:
        """
        Fetch movie data from OMDb API (includes RT scores).

        Args:
            title: Movie title

        Returns:
            OMDb data dict or None
        """
        if not self.omdb_api_key:
            return None

        cache_key = f"omdb_{title}"
        cached = self._load_cache(cache_key)
        if cached:
            return cached

        try:
            response = self.session.get(
                self.OMDB_API,
                params={
                    'apikey': self.omdb_api_key,
                    't': title,
                    'type': 'movie'
                },
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data.get('Response') == 'True':
                    self._save_cache(cache_key, data)
                    return data

        except Exception as e:
            logger.debug(f"OMDb error for {title}: {e}")

        return None

    def get_rt_score(self, title: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Get Rotten Tomatoes scores for a movie.

        Args:
            title: Movie title

        Returns:
            Tuple of (critic_score, audience_score)
        """
        omdb_data = self.fetch_omdb_data(title)

        if not omdb_data:
            return None, None

        critic_score = None
        audience_score = None

        # Parse ratings from OMDb
        for rating in omdb_data.get('Ratings', []):
            source = rating.get('Source', '')
            value = rating.get('Value', '')

            if 'Rotten Tomatoes' in source:
                try:
                    critic_score = int(value.replace('%', ''))
                except Exception as e:
                    logger.debug(f"Error parsing RT critic score: {e}")

        # Metascore can approximate audience sentiment
        try:
            metascore = int(omdb_data.get('Metascore', 0))
            if metascore > 0:
                audience_score = metascore
        except Exception as e:
            logger.debug(f"Error parsing metascore: {e}")

        return critic_score, audience_score

    def estimate_opening_weekend(
        self,
        title: str,
        budget: Optional[float] = None,
        franchise: Optional[str] = None,
        rt_score: Optional[int] = None,
        theaters: Optional[int] = None
    ) -> Tuple[float, float, str]:
        """
        Estimate opening weekend gross based on various factors.

        Args:
            title: Movie title
            budget: Production budget in millions
            franchise: Franchise name if applicable
            rt_score: Rotten Tomatoes critic score
            theaters: Number of theaters

        Returns:
            Tuple of (estimate_low, estimate_high, reasoning)
        """
        # Base estimate from budget (typically opens at 30-50% of budget)
        if budget:
            base_estimate = budget * 0.4
        elif theaters:
            # Estimate from theater count ($5k-15k per theater)
            base_estimate = theaters * 0.01  # $10k avg per theater in millions
        else:
            base_estimate = 30.0  # Default $30M for major release

        # Apply franchise multiplier
        franchise_mult = 1.0
        if franchise:
            franchise_mult = FRANCHISE_MULTIPLIERS.get(franchise, FRANCHISE_MULTIPLIERS['default'])

        # Apply RT score boost
        rt_mult = 1.0
        if rt_score:
            for (low, high), mult in RT_SCORE_BOOST.items():
                if low <= rt_score < high:
                    rt_mult = mult
                    break

        # Calculate estimate with variance
        estimate = base_estimate * franchise_mult * rt_mult

        # Wide range due to uncertainty
        estimate_low = estimate * 0.7
        estimate_high = estimate * 1.3

        reasoning_parts = [f"Base: ${base_estimate:.0f}M"]
        if franchise:
            reasoning_parts.append(f"Franchise ({franchise}): {franchise_mult:.2f}x")
        if rt_score:
            reasoning_parts.append(f"RT ({rt_score}%): {rt_mult:.2f}x")

        reasoning = ", ".join(reasoning_parts)

        return estimate_low, estimate_high, reasoning

    def calculate_threshold_probability(
        self,
        estimate_low: float,
        estimate_high: float,
        threshold: float
    ) -> float:
        """
        Calculate probability of exceeding a threshold.

        Uses a simple uniform distribution between low and high estimates.

        Args:
            estimate_low: Low estimate in millions
            estimate_high: High estimate in millions
            threshold: Threshold in millions

        Returns:
            Probability of exceeding threshold (0-1)
        """
        if threshold <= estimate_low:
            return 0.95  # Very likely above
        elif threshold >= estimate_high:
            return 0.05  # Very unlikely above
        else:
            # Linear interpolation
            range_size = estimate_high - estimate_low
            position = threshold - estimate_low
            prob_below = position / range_size
            return 1 - prob_below

    def get_movie_data(self, title: str, **kwargs) -> MovieData:
        """
        Get comprehensive movie data.

        Args:
            title: Movie title
            **kwargs: Additional movie info

        Returns:
            MovieData object
        """
        rt_critic, rt_audience = self.get_rt_score(title)

        # Detect franchise
        franchise = kwargs.get('franchise')
        is_sequel = False

        for f in FRANCHISE_MULTIPLIERS.keys():
            if f.lower() in title.lower():
                franchise = f
                is_sequel = True
                break

        if any(x in title for x in ['2', '3', '4', 'II', 'III', 'IV', 'Part']):
            is_sequel = True

        return MovieData(
            title=title,
            release_date=kwargs.get('release_date'),
            distributor=kwargs.get('distributor'),
            budget=kwargs.get('budget'),
            opening_weekend=kwargs.get('opening_weekend'),
            total_gross=kwargs.get('total_gross'),
            theaters=kwargs.get('theaters'),
            rt_score=rt_critic,
            rt_audience=rt_audience,
            metacritic=None,
            genre=kwargs.get('genre'),
            is_sequel=is_sequel,
            franchise=franchise,
            source='BoxOfficeScraper',
            fetched_at=datetime.now(timezone.utc)
        )

    def generate_probability_estimates(
        self,
        movies: List[MovieData]
    ) -> List[BoxOfficeProbability]:
        """
        Generate probability estimates for Kalshi box office contracts.

        Args:
            movies: List of MovieData

        Returns:
            List of BoxOfficeProbability estimates
        """
        estimates = []

        # Common Kalshi thresholds (in millions)
        thresholds = [25, 50, 75, 100, 150, 200, 250, 300]

        for movie in movies:
            estimate_low, estimate_high, reasoning = self.estimate_opening_weekend(
                movie.title,
                budget=movie.budget,
                franchise=movie.franchise,
                rt_score=movie.rt_score,
                theaters=movie.theaters
            )

            # Generate estimates for relevant thresholds
            midpoint = (estimate_low + estimate_high) / 2

            for threshold in thresholds:
                # Only generate for thresholds near the estimate
                if threshold < estimate_low * 0.5 or threshold > estimate_high * 2:
                    continue

                prob = self.calculate_threshold_probability(estimate_low, estimate_high, threshold)

                # Above threshold
                estimates.append(BoxOfficeProbability(
                    movie_title=movie.title,
                    ticker_pattern=f'KXBOX-{movie.title[:10].upper().replace(" ", "")}-O{threshold}',
                    threshold=threshold,
                    direction='above',
                    our_probability=prob,
                    reasoning=f"Est: ${estimate_low:.0f}M-${estimate_high:.0f}M. {reasoning}",
                    confidence='LOW' if not movie.rt_score else 'MEDIUM'
                ))

        logger.info(f"Generated {len(estimates)} box office probability estimates")
        return estimates


def main():
    """Test the box office scraper"""
    print("=" * 60)
    print("BOX OFFICE SCRAPER TEST")
    print("=" * 60)

    scraper = BoxOfficeScraper()

    print("\n[1] Fetching Weekend Box Office...")
    print("-" * 40)
    weekend = scraper.fetch_bom_weekend()

    print(f"Top 5 movies this weekend:")
    for movie in weekend[:5]:
        print(f"  {movie.get('title')}")
        if movie.get('weekend_gross'):
            print(f"    Weekend: ${movie['weekend_gross']:.1f}M")
        if movie.get('total_gross'):
            print(f"    Total: ${movie['total_gross']:.1f}M")

    print("\n[2] Upcoming Releases...")
    print("-" * 40)
    upcoming = scraper.fetch_upcoming_releases()

    print(f"Found {len(upcoming)} upcoming releases:")
    for movie in upcoming[:5]:
        print(f"  {movie.get('title')} - {movie.get('release_date', 'TBD')}")

    print("\n[3] Opening Weekend Estimates...")
    print("-" * 40)

    test_movies = [
        {'title': 'Captain America: Brave New World', 'franchise': 'Marvel', 'budget': 180},
        {'title': 'Snow White', 'franchise': 'Disney', 'budget': 200},
        {'title': 'Superman', 'franchise': 'DC', 'budget': 250},
    ]

    movie_data = []
    for m in test_movies:
        md = scraper.get_movie_data(**m)
        movie_data.append(md)

        est_low, est_high, reasoning = scraper.estimate_opening_weekend(
            m['title'],
            budget=m.get('budget'),
            franchise=m.get('franchise'),
            rt_score=md.rt_score
        )

        print(f"\n{m['title']}:")
        print(f"  Budget: ${m.get('budget', 'N/A')}M")
        print(f"  RT Score: {md.rt_score or 'N/A'}")
        print(f"  Opening Est: ${est_low:.0f}M - ${est_high:.0f}M")

    print("\n[4] Probability Estimates...")
    print("-" * 40)

    estimates = scraper.generate_probability_estimates(movie_data)

    for est in estimates[:10]:
        print(f"  {est.movie_title}: >${est.threshold}M = {est.our_probability:.0%}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
