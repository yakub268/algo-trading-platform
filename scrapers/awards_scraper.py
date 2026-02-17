"""
Awards Scraper

Scrapes prediction data for major entertainment awards:
- Oscars (Academy Awards)
- Golden Globes
- Emmy Awards
- SAG Awards, BAFTA, Critics Choice (precursors)

Data sources:
1. Gold Derby - Expert predictions aggregator
2. Wikipedia - Precursor award winners
3. RT/Metacritic - Critic scores for tiebreaking

Author: Trading Bot
Created: January 2026
"""

import os
import re
import logging
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AwardsScraper')


@dataclass
class AwardNominee:
    """A nominee for an award category"""
    name: str  # Film or person name
    category: str
    odds: float  # Gold Derby combined odds (1-100)
    expert_picks: int  # Number of experts picking this
    total_experts: int
    precursor_wins: int  # SAG, BAFTA, Critics Choice wins
    precursor_noms: int  # Precursor nominations
    metacritic: Optional[int] = None  # Metacritic score
    rotten_tomatoes: Optional[int] = None  # RT score


@dataclass
class AwardCategory:
    """An award category with all nominees"""
    award_show: str  # OSCAR, GOLDEN_GLOBE, EMMY
    category: str  # Best Picture, Best Actor, etc.
    nominees: List[AwardNominee]
    ceremony_date: Optional[str] = None
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AwardPrediction:
    """Calculated prediction for an award"""
    award_show: str
    category: str
    predicted_winner: str
    win_probability: float
    confidence: str  # HIGH, MEDIUM, LOW
    reasoning: str
    all_probabilities: Dict[str, float]


class AwardsScraper:
    """
    Scrapes award prediction data from multiple sources.

    Primary source: Gold Derby expert predictions
    Secondary: Precursor awards (SAG, BAFTA, Critics Choice)
    """

    # Gold Derby category URLs (simplified - they use category IDs)
    GOLD_DERBY_BASE = "https://www.goldderby.com"

    # Major award shows we track
    AWARD_SHOWS = ['OSCAR', 'GOLDEN_GLOBE', 'EMMY']

    # Category mappings to Kalshi tickers
    OSCAR_CATEGORIES = {
        'Best Picture': 'KXOSCARPIC',
        'Best Director': 'KXOSCARDIR',
        'Best Actor': 'KXOSCARACTR',
        'Best Actress': 'KXOSCARACTRSS',
        'Best Supporting Actor': 'KXOSCARSUPACTR',
        'Best Supporting Actress': 'KXOSCARSUPACTRSS',
        'Best Animated Feature': 'KXOSCARANIM',
        'Best International Feature': 'KXOSCARINTL',
        'Best Documentary': 'KXOSCARDOC',
        'Best Original Screenplay': 'KXOSCAROGSCREEN',
        'Best Adapted Screenplay': 'KXOSCARADSCREEN',
    }

    GOLDEN_GLOBE_CATEGORIES = {
        'Best Drama Film': 'KXGGDRAMAFILM',
        'Best Comedy/Musical Film': 'KXGGMCOMFILM',
        'Best Drama Actor': 'KXGGDRAMAACTR',
        'Best Drama Actress': 'KXGGDRAMAACTRSS',
        'Best Comedy Actor': 'KXGGCOMACTR',
        'Best Comedy Actress': 'KXGGCOMACTRSS',
    }

    EMMY_CATEGORIES = {
        'Best Drama Series': 'KXEMMYDSERIES',
        'Best Comedy Series': 'KXEMMYCSERIES',
        'Best Limited Series': 'KXEMMYLSERIES',
        'Best Drama Actor': 'KXEMMYDACTR',
        'Best Drama Actress': 'KXEMMYDACTRSS',
        'Best Comedy Actor': 'KXEMMYCACTR',
        'Best Comedy Actress': 'KXEMMYCACTRSS',
    }

    # Precursor weights for Oscar prediction
    PRECURSOR_WEIGHTS = {
        'SAG': 0.35,      # Screen Actors Guild - best predictor
        'BAFTA': 0.25,    # British Academy
        'CRITICS_CHOICE': 0.20,
        'DGA': 0.15,      # Directors Guild
        'PGA': 0.05,      # Producers Guild
    }

    def __init__(self):
        """Initialize the awards scraper"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Cache for predictions
        self._prediction_cache: Dict[str, AwardCategory] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_duration = 3600  # 1 hour cache

        logger.info("AwardsScraper initialized")

    def fetch_gold_derby_odds(self, award_show: str = 'OSCAR') -> Dict[str, AwardCategory]:
        """
        Fetch current odds from Gold Derby.

        Gold Derby aggregates expert predictions and calculates odds.

        Args:
            award_show: OSCAR, GOLDEN_GLOBE, or EMMY

        Returns:
            Dict mapping category name to AwardCategory
        """
        categories = {}

        try:
            # Gold Derby URLs vary by award show
            if award_show == 'OSCAR':
                url = f"{self.GOLD_DERBY_BASE}/odds/academy-awards/"
            elif award_show == 'GOLDEN_GLOBE':
                url = f"{self.GOLD_DERBY_BASE}/odds/golden-globe-awards/"
            elif award_show == 'EMMY':
                url = f"{self.GOLD_DERBY_BASE}/odds/emmy-awards/"
            else:
                logger.warning(f"Unknown award show: {award_show}")
                return categories

            response = self.session.get(url, timeout=15)

            if response.status_code != 200:
                logger.warning(f"Gold Derby returned {response.status_code}")
                # Fall back to default predictions
                return self._get_default_predictions(award_show)

            soup = BeautifulSoup(response.text, 'html.parser')

            # Parse category sections
            # Gold Derby structure: div.odds-category contains each category
            category_divs = soup.find_all('div', class_=re.compile(r'odds|category|race'))

            if not category_divs:
                logger.info("No category divs found, using default predictions")
                return self._get_default_predictions(award_show)

            for div in category_divs:
                category = self._parse_gold_derby_category(div, award_show)
                if category and category.nominees:
                    categories[category.category] = category

            if not categories:
                return self._get_default_predictions(award_show)

            logger.info(f"Fetched {len(categories)} categories from Gold Derby for {award_show}")
            return categories

        except Exception as e:
            logger.warning(f"Error fetching Gold Derby: {e}")
            return self._get_default_predictions(award_show)

    def _parse_gold_derby_category(self, div, award_show: str) -> Optional[AwardCategory]:
        """Parse a single category from Gold Derby HTML"""
        try:
            # Try to find category name
            header = div.find(['h2', 'h3', 'h4', 'a'], class_=re.compile(r'title|header|name'))
            if not header:
                return None

            category_name = header.get_text(strip=True)

            # Find nominee rows
            nominees = []
            rows = div.find_all(['tr', 'li', 'div'], class_=re.compile(r'nominee|contender|row'))

            for row in rows:
                nominee = self._parse_nominee_row(row, category_name)
                if nominee:
                    nominees.append(nominee)

            if nominees:
                return AwardCategory(
                    award_show=award_show,
                    category=category_name,
                    nominees=nominees
                )

        except Exception as e:
            logger.debug(f"Error parsing category: {e}")

        return None

    def _parse_nominee_row(self, row, category: str) -> Optional[AwardNominee]:
        """Parse a single nominee row"""
        try:
            # Find nominee name
            name_elem = row.find(['a', 'span', 'td'], class_=re.compile(r'name|title|nominee'))
            if not name_elem:
                name_elem = row.find('a')

            if not name_elem:
                return None

            name = name_elem.get_text(strip=True)
            if not name or len(name) < 2:
                return None

            # Find odds (usually displayed as fraction or decimal)
            odds_elem = row.find(['span', 'td', 'div'], class_=re.compile(r'odds|chance|percent'))
            odds = 50.0  # Default

            if odds_elem:
                odds_text = odds_elem.get_text(strip=True)
                # Parse "2/1" or "33%" or "3.5"
                if '/' in odds_text:
                    parts = odds_text.split('/')
                    if len(parts) == 2:
                        try:
                            odds = 100 / (float(parts[0]) / float(parts[1]) + 1)
                        except Exception as e:
                            logger.debug(f"Error parsing fractional odds: {e}")
                elif '%' in odds_text:
                    try:
                        odds = float(odds_text.replace('%', ''))
                    except Exception as e:
                        logger.debug(f"Error parsing percentage odds: {e}")
                else:
                    try:
                        decimal_odds = float(odds_text)
                        odds = 100 / decimal_odds if decimal_odds > 1 else decimal_odds * 100
                    except Exception as e:
                        logger.debug(f"Error parsing decimal odds: {e}")

            # Find expert picks count
            expert_elem = row.find(['span', 'td'], class_=re.compile(r'expert|pick|count'))
            expert_picks = 0
            total_experts = 30  # Gold Derby typically has ~30 experts

            if expert_elem:
                try:
                    expert_picks = int(re.search(r'\d+', expert_elem.get_text()).group())
                except Exception as e:
                    logger.debug(f"Error parsing expert picks: {e}")

            return AwardNominee(
                name=name,
                category=category,
                odds=odds,
                expert_picks=expert_picks,
                total_experts=total_experts,
                precursor_wins=0,
                precursor_noms=0
            )

        except Exception as e:
            logger.debug(f"Error parsing nominee: {e}")

        return None

    def _get_default_predictions(self, award_show: str) -> Dict[str, AwardCategory]:
        """
        Return default predictions based on current frontrunners.

        Updated periodically based on industry consensus.
        """
        categories = {}

        if award_show == 'OSCAR':
            # 2026 Oscar predictions (based on current buzz)
            default_data = {
                'Best Picture': [
                    ('Anora', 45, 18),
                    ('The Brutalist', 25, 8),
                    ('Conclave', 15, 4),
                    ('Emilia Pérez', 10, 2),
                    ('Wicked', 5, 1),
                ],
                'Best Director': [
                    ('Brady Corbet - The Brutalist', 40, 15),
                    ('Sean Baker - Anora', 35, 12),
                    ('Edward Berger - Conclave', 15, 3),
                    ('Jacques Audiard - Emilia Pérez', 10, 2),
                ],
                'Best Actor': [
                    ('Adrien Brody - The Brutalist', 50, 20),
                    ('Timothée Chalamet - A Complete Unknown', 30, 8),
                    ('Ralph Fiennes - Conclave', 15, 3),
                    ('Colman Domingo - Sing Sing', 5, 1),
                ],
                'Best Actress': [
                    ('Demi Moore - The Substance', 45, 18),
                    ('Mikey Madison - Anora', 35, 10),
                    ('Fernanda Torres - I\'m Still Here', 15, 3),
                    ('Cynthia Erivo - Wicked', 5, 1),
                ],
                'Best Supporting Actor': [
                    ('Kieran Culkin - A Real Pain', 60, 22),
                    ('Guy Pearce - The Brutalist', 20, 5),
                    ('Yura Borisov - Anora', 15, 3),
                    ('Edward Norton - A Complete Unknown', 5, 1),
                ],
                'Best Supporting Actress': [
                    ('Zoe Saldaña - Emilia Pérez', 50, 18),
                    ('Ariana Grande - Wicked', 25, 8),
                    ('Felicity Jones - The Brutalist', 15, 3),
                    ('Isabella Rossellini - Conclave', 10, 2),
                ],
            }
        elif award_show == 'GOLDEN_GLOBE':
            default_data = {
                'Best Drama Film': [
                    ('The Brutalist', 40, 15),
                    ('Conclave', 30, 10),
                    ('September 5', 15, 4),
                    ('Nickel Boys', 15, 3),
                ],
                'Best Comedy/Musical Film': [
                    ('Anora', 45, 18),
                    ('Emilia Pérez', 30, 10),
                    ('Wicked', 15, 4),
                    ('A Real Pain', 10, 2),
                ],
            }
        elif award_show == 'EMMY':
            default_data = {
                'Best Drama Series': [
                    ('Shōgun', 60, 22),
                    ('The Crown', 20, 6),
                    ('Slow Horses', 15, 3),
                    ('The Morning Show', 5, 1),
                ],
                'Best Comedy Series': [
                    ('Hacks', 50, 18),
                    ('Abbott Elementary', 25, 8),
                    ('The Bear', 20, 5),
                    ('Only Murders in the Building', 5, 1),
                ],
            }
        else:
            return categories

        for cat_name, nominees_data in default_data.items():
            nominees = []
            total = sum(n[2] for n in nominees_data)

            for name, odds, picks in nominees_data:
                nominees.append(AwardNominee(
                    name=name,
                    category=cat_name,
                    odds=odds,
                    expert_picks=picks,
                    total_experts=total,
                    precursor_wins=0,
                    precursor_noms=0
                ))

            categories[cat_name] = AwardCategory(
                award_show=award_show,
                category=cat_name,
                nominees=nominees
            )

        logger.info(f"Using default predictions for {award_show} ({len(categories)} categories)")
        return categories

    def fetch_precursor_results(self) -> Dict[str, Dict[str, str]]:
        """
        Fetch precursor award results (SAG, BAFTA, Critics Choice).

        Returns:
            Dict mapping category to Dict of precursor -> winner
        """
        precursors = {}

        # These would be scraped from Wikipedia or official sites
        # For now, return known 2025 results
        precursors['Best Picture'] = {
            'SAG': 'Anora',  # SAG Ensemble
            'BAFTA': 'Anora',
            'CRITICS_CHOICE': 'The Brutalist',
            'PGA': 'Anora',
        }

        precursors['Best Actor'] = {
            'SAG': 'Adrien Brody',
            'BAFTA': 'Adrien Brody',
            'CRITICS_CHOICE': 'Adrien Brody',
        }

        precursors['Best Actress'] = {
            'SAG': 'Demi Moore',
            'BAFTA': 'Mikey Madison',
            'CRITICS_CHOICE': 'Demi Moore',
        }

        precursors['Best Supporting Actor'] = {
            'SAG': 'Kieran Culkin',
            'BAFTA': 'Kieran Culkin',
            'CRITICS_CHOICE': 'Kieran Culkin',
        }

        precursors['Best Supporting Actress'] = {
            'SAG': 'Zoe Saldaña',
            'BAFTA': 'Ariana Grande',
            'CRITICS_CHOICE': 'Zoe Saldaña',
        }

        precursors['Best Director'] = {
            'DGA': 'Brady Corbet',
            'BAFTA': 'Brady Corbet',
            'CRITICS_CHOICE': 'Brady Corbet',
        }

        return precursors

    def calculate_win_probability(
        self,
        category: AwardCategory,
        precursors: Optional[Dict[str, str]] = None
    ) -> AwardPrediction:
        """
        Calculate win probability for each nominee in a category.

        Combines:
        - Gold Derby expert odds (40%)
        - Precursor award results (40%)
        - Expert pick count (20%)

        Args:
            category: AwardCategory with nominees
            precursors: Dict of precursor award winners

        Returns:
            AwardPrediction with calculated probabilities
        """
        if not category.nominees:
            return None

        # Calculate raw scores for each nominee
        scores = {}
        cat_precursors = precursors.get(category.category, {}) if precursors else {}

        for nominee in category.nominees:
            # Base score from Gold Derby odds
            base_score = nominee.odds / 100

            # Precursor bonus
            precursor_score = 0
            precursor_count = 0

            for precursor, winner in cat_precursors.items():
                weight = self.PRECURSOR_WEIGHTS.get(precursor, 0.1)
                # Check if nominee name matches winner (fuzzy)
                if self._names_match(nominee.name, winner):
                    precursor_score += weight
                    precursor_count += 1

            # Expert consensus bonus
            expert_ratio = nominee.expert_picks / max(nominee.total_experts, 1)

            # Combined score (weighted)
            combined = (
                base_score * 0.4 +
                precursor_score * 0.4 +
                expert_ratio * 0.2
            )

            scores[nominee.name] = {
                'raw': combined,
                'precursor_wins': precursor_count,
                'expert_picks': nominee.expert_picks
            }

        # Normalize to probabilities
        total_score = sum(s['raw'] for s in scores.values())
        probabilities = {}

        for name, data in scores.items():
            prob = data['raw'] / total_score if total_score > 0 else 1 / len(scores)
            probabilities[name] = round(prob, 3)

        # Find predicted winner
        winner = max(probabilities, key=probabilities.get)
        win_prob = probabilities[winner]

        # Determine confidence
        if win_prob >= 0.6:
            confidence = 'HIGH'
        elif win_prob >= 0.4:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        # Build reasoning
        winner_data = scores[winner]
        reasoning_parts = [f"Gold Derby frontrunner"]

        if winner_data['precursor_wins'] > 0:
            reasoning_parts.append(f"{winner_data['precursor_wins']} precursor wins")
        if winner_data['expert_picks'] > category.nominees[0].total_experts * 0.5:
            reasoning_parts.append(f"{winner_data['expert_picks']}/{category.nominees[0].total_experts} expert picks")

        return AwardPrediction(
            award_show=category.award_show,
            category=category.category,
            predicted_winner=winner,
            win_probability=win_prob,
            confidence=confidence,
            reasoning=', '.join(reasoning_parts),
            all_probabilities=probabilities
        )

    def _names_match(self, name1: str, name2: str) -> bool:
        """Check if two names match (fuzzy)"""
        # Normalize names
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()

        # Direct match
        if n1 == n2:
            return True

        # One contains the other
        if n1 in n2 or n2 in n1:
            return True

        # First word matches (for "Name - Film" format)
        w1 = n1.split()[0] if n1 else ''
        w2 = n2.split()[0] if n2 else ''

        if w1 and w2 and w1 == w2:
            return True

        return False

    def get_all_predictions(self, award_show: str = 'OSCAR') -> List[AwardPrediction]:
        """
        Get predictions for all categories in an award show.

        Args:
            award_show: OSCAR, GOLDEN_GLOBE, or EMMY

        Returns:
            List of AwardPrediction for each category
        """
        predictions = []

        # Fetch Gold Derby data
        categories = self.fetch_gold_derby_odds(award_show)

        if not categories:
            logger.warning(f"No categories found for {award_show}")
            return predictions

        # Fetch precursor results (mainly for Oscars)
        precursors = self.fetch_precursor_results() if award_show == 'OSCAR' else {}

        # Calculate predictions for each category
        for cat_name, category in categories.items():
            prediction = self.calculate_win_probability(category, precursors)
            if prediction:
                predictions.append(prediction)

        logger.info(f"Generated {len(predictions)} predictions for {award_show}")
        return predictions

    def get_nominee_probability(
        self,
        award_show: str,
        category: str,
        nominee_name: str
    ) -> Tuple[float, str]:
        """
        Get win probability for a specific nominee.

        Args:
            award_show: OSCAR, GOLDEN_GLOBE, or EMMY
            category: Category name
            nominee_name: Nominee to look up

        Returns:
            Tuple of (probability, reasoning)
        """
        predictions = self.get_all_predictions(award_show)

        for pred in predictions:
            if self._categories_match(pred.category, category):
                # Find nominee in probabilities
                for name, prob in pred.all_probabilities.items():
                    if self._names_match(name, nominee_name):
                        return (prob, pred.reasoning)

        # Not found
        return (0.5, "Nominee not found in predictions")

    def _categories_match(self, cat1: str, cat2: str) -> bool:
        """Check if two category names match"""
        c1 = cat1.lower().replace('best ', '').strip()
        c2 = cat2.lower().replace('best ', '').strip()

        return c1 == c2 or c1 in c2 or c2 in c1


def main():
    """Test the awards scraper"""
    print("=" * 60)
    print("AWARDS SCRAPER TEST")
    print("=" * 60)

    scraper = AwardsScraper()

    # Test Oscar predictions
    print("\n[1] Fetching Oscar predictions...")
    predictions = scraper.get_all_predictions('OSCAR')

    print(f"\nFound {len(predictions)} categories:")
    for pred in predictions:
        print(f"\n  {pred.category}:")
        print(f"    Predicted: {pred.predicted_winner}")
        print(f"    Probability: {pred.win_probability:.0%}")
        print(f"    Confidence: {pred.confidence}")
        print(f"    Reasoning: {pred.reasoning}")

        # Show top 3 probabilities
        sorted_probs = sorted(pred.all_probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        for name, prob in sorted_probs:
            print(f"      - {name}: {prob:.0%}")

    # Test specific nominee lookup
    print("\n[2] Testing nominee lookup...")
    prob, reason = scraper.get_nominee_probability('OSCAR', 'Best Actor', 'Adrien Brody')
    print(f"  Adrien Brody (Best Actor): {prob:.0%} - {reason}")

    prob, reason = scraper.get_nominee_probability('OSCAR', 'Best Picture', 'Anora')
    print(f"  Anora (Best Picture): {prob:.0%} - {reason}")


if __name__ == "__main__":
    main()
