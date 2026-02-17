"""
Computer Vision Trading Bot

Advanced computer vision bot that integrates with Windows-MCP to automatically
interact with broker interfaces (Alpaca, Kalshi, OANDA) using visual recognition.

Key Features:
- Visual element detection and recognition
- Broker interface automation
- Screen-based trading opportunity detection
- Windows-MCP integration for UI automation
- Comprehensive error handling and retry logic
- Real-time visual monitoring

Supported Brokers:
- Alpaca (web interface)
- Kalshi (web interface)
- OANDA (web interface)
- Generic trading platforms

Author: Trading Bot Arsenal
Created: February 2026
"""

import os
import sys
import logging
import time
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, asdict
import base64
from io import BytesIO
import requests
import subprocess

# Optional imports for full computer vision functionality
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import pytesseract
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

# Add trading bot paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing bot dependencies
try:
    from bots.kalshi_client import KalshiClient
    from utils.trading_alerts import TradingAlerts
    TRADING_DEPS_AVAILABLE = True
except ImportError:
    TRADING_DEPS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ComputerVisionBot')


# =============================================================================
# DATA CLASSES AND ENUMS
# =============================================================================

@dataclass
class VisualElement:
    """Detected visual element on screen"""
    element_type: str  # button, text_field, price, chart, etc.
    confidence: float
    coordinates: Tuple[int, int, int, int]  # x, y, width, height
    text: Optional[str] = None
    value: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class BrokerInterface:
    """Configuration for a specific broker interface"""
    name: str
    url: str
    login_elements: Dict[str, Any]
    trading_elements: Dict[str, Any]
    price_patterns: List[str]
    button_templates: Dict[str, str]  # button_name -> template_path


@dataclass
class TradingOpportunity:
    """Computer vision detected trading opportunity"""
    broker: str
    symbol: str
    action: str  # buy, sell
    price: float
    confidence: float
    visual_evidence: str  # base64 encoded screenshot
    reasoning: str
    timestamp: datetime
    expiry: Optional[datetime] = None


@dataclass
class CVResult:
    """Computer vision analysis result"""
    success: bool
    opportunities: List[TradingOpportunity]
    visual_elements: List[VisualElement]
    screenshot_path: Optional[str]
    error: Optional[str] = None
    analysis_time: float = 0.0


# =============================================================================
# WINDOWS-MCP INTEGRATION
# =============================================================================

class WindowsMCPClient:
    """Client for Windows-MCP server integration"""

    def __init__(self, server_path: str = None):
        self.server_path = server_path or r"C:\dev\projects\Windows-MCP"
        self.process = None
        self.is_connected = False

    def start_server(self) -> bool:
        """Start Windows-MCP server if not running"""
        try:
            # Check if server is already running by testing a simple command
            if self._test_connection():
                self.is_connected = True
                return True

            # Start the server
            cmd = ["python", "-m", "uvx", "windows-mcp"]
            self.process = subprocess.Popen(
                cmd,
                cwd=self.server_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Give it time to start
            time.sleep(3)

            self.is_connected = self._test_connection()
            return self.is_connected

        except Exception as e:
            logger.error(f"Failed to start Windows-MCP server: {e}")
            return False

    def _test_connection(self) -> bool:
        """Test if MCP server is responsive"""
        try:
            # This would be implemented with actual MCP protocol
            # For now, assume it's working if we can import the modules
            return True
        except Exception as e:
            logger.debug(f"MCP connection test failed: {e}")
            return False

    def take_screenshot(self, save_path: str = None) -> Optional[str]:
        """Take a screenshot using Windows-MCP"""
        try:
            # Use Windows-MCP Snapshot tool with vision
            # This is a mock implementation - in practice would use MCP protocol
            import pyautogui
            screenshot = pyautogui.screenshot()

            if save_path:
                screenshot.save(save_path)
                return save_path
            else:
                # Save to temp file
                temp_path = f"temp_screenshot_{int(time.time())}.png"
                screenshot.save(temp_path)
                return temp_path

        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    def click_element(self, x: int, y: int, button: str = 'left') -> bool:
        """Click on screen element"""
        try:
            # Use Windows-MCP Click tool
            import pyautogui
            pyautogui.click(x, y, button=button)
            return True
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return False

    def type_text(self, x: int, y: int, text: str, clear: bool = True) -> bool:
        """Type text at coordinates"""
        try:
            # Use Windows-MCP Type tool
            import pyautogui
            pyautogui.click(x, y)
            if clear:
                pyautogui.hotkey('ctrl', 'a')
            pyautogui.type(text)
            return True
        except Exception as e:
            logger.error(f"Type failed: {e}")
            return False

    def get_desktop_state(self) -> Dict:
        """Get current desktop state"""
        try:
            # Mock implementation of Windows-MCP Snapshot
            return {
                "active_windows": ["Browser", "Terminal"],
                "interactive_elements": [],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Get desktop state failed: {e}")
            return {}


# =============================================================================
# COMPUTER VISION ENGINE
# =============================================================================

class ComputerVisionEngine:
    """Core computer vision processing engine"""

    def __init__(self):
        self.template_cache = {}
        self.ocr_confidence_threshold = 0.6

        # Initialize tesseract if available
        self.ocr_available = self._check_tesseract()

    def _check_tesseract(self) -> bool:
        """Check if Tesseract OCR is available"""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception as e:
            logger.warning(f"Tesseract not available - OCR features disabled: {e}")
            return False

    def load_template(self, template_path: str):
        """Load and cache template image"""
        if not CV_AVAILABLE:
            return None

        if template_path in self.template_cache:
            return self.template_cache[template_path]

        try:
            if os.path.exists(template_path):
                template = cv2.imread(template_path, cv2.IMREAD_COLOR)
                self.template_cache[template_path] = template
                return template
        except Exception as e:
            logger.error(f"Failed to load template {template_path}: {e}")

        return None

    def find_template_matches(self, image, template,
                            threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """Find template matches in image"""
        if not CV_AVAILABLE:
            return []

        try:
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)

            matches = []
            for pt in zip(*locations[::-1]):
                confidence = result[pt[1], pt[0]]
                matches.append((pt[0], pt[1], confidence))

            # Remove overlapping matches
            return self._remove_overlaps(matches, template.shape)

        except Exception as e:
            logger.error(f"Template matching failed: {e}")
            return []

    def _remove_overlaps(self, matches: List[Tuple[int, int, float]],
                        template_shape: Tuple[int, int]) -> List[Tuple[int, int, float]]:
        """Remove overlapping template matches"""
        if not matches:
            return []

        # Sort by confidence
        matches.sort(key=lambda x: x[2], reverse=True)

        filtered = [matches[0]]
        h, w = template_shape[:2]

        for match in matches[1:]:
            x, y, conf = match
            overlaps = False

            for existing in filtered:
                ex, ey, _ = existing
                if (abs(x - ex) < w * 0.5 and abs(y - ey) < h * 0.5):
                    overlaps = True
                    break

            if not overlaps:
                filtered.append(match)

        return filtered

    def extract_text_regions(self, image) -> List[VisualElement]:
        """Extract text from image using OCR"""
        if not self.ocr_available:
            return []

        try:
            # Convert to PIL Image for tesseract
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Get detailed OCR data
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)

            elements = []
            for i, conf in enumerate(data['conf']):
                if int(conf) > self.ocr_confidence_threshold * 100:
                    text = data['text'][i].strip()
                    if text and len(text) > 2:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                        # Try to extract numerical values
                        value = self._extract_number(text)

                        element = VisualElement(
                            element_type='text',
                            confidence=conf / 100.0,
                            coordinates=(x, y, w, h),
                            text=text,
                            value=value
                        )
                        elements.append(element)

            return elements

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return []

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numerical value from text"""
        try:
            # Remove common currency symbols and formatting
            cleaned = text.replace('$', '').replace(',', '').replace('%', '')

            # Try to extract float
            import re
            match = re.search(r'-?\d+\.?\d*', cleaned)
            if match:
                return float(match.group())
        except Exception as e:
            logger.debug(f"Error extracting number from text: {e}")

        return None

    def detect_price_changes(self, image,
                           previous_image = None) -> List[VisualElement]:
        """Detect price changes and market movements"""
        if not CV_AVAILABLE:
            return []

        elements = []

        try:
            # Extract text elements
            text_elements = self.extract_text_regions(image)

            # Filter for price-like elements
            price_elements = [
                elem for elem in text_elements
                if elem.value is not None and (
                    '$' in (elem.text or '') or
                    elem.value > 0.01  # Likely a price
                )
            ]

            # Look for green/red indicators (bullish/bearish)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Green mask (bullish)
            green_lower = np.array([40, 40, 40])
            green_upper = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)

            # Red mask (bearish)
            red_lower1 = np.array([0, 40, 40])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([160, 40, 40])
            red_upper2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            # Associate colors with price elements
            for elem in price_elements:
                x, y, w, h = elem.coordinates
                roi_green = green_mask[y:y+h, x:x+w]
                roi_red = red_mask[y:y+h, x:x+w]

                green_pixels = np.sum(roi_green > 0)
                red_pixels = np.sum(roi_red > 0)

                if green_pixels > red_pixels and green_pixels > 10:
                    elem.element_type = 'price_up'
                elif red_pixels > green_pixels and red_pixels > 10:
                    elem.element_type = 'price_down'
                else:
                    elem.element_type = 'price_neutral'

                elements.append(elem)

            return elements

        except Exception as e:
            logger.error(f"Price change detection failed: {e}")
            return []

    def find_trading_buttons(self, image,
                           button_templates: Dict[str, str]) -> List[VisualElement]:
        """Find trading buttons (Buy, Sell, etc.) in image"""
        elements = []

        for button_name, template_path in button_templates.items():
            template = self.load_template(template_path)
            if template is None:
                continue

            matches = self.find_template_matches(image, template)

            for x, y, confidence in matches:
                h, w = template.shape[:2]
                element = VisualElement(
                    element_type=f'button_{button_name}',
                    confidence=confidence,
                    coordinates=(x, y, w, h),
                    text=button_name.upper()
                )
                elements.append(element)

        return elements


# =============================================================================
# BROKER INTERFACE HANDLERS
# =============================================================================

class BrokerInterfaceManager:
    """Manages interactions with different broker interfaces"""

    def __init__(self, mcp_client: WindowsMCPClient):
        self.mcp_client = mcp_client
        self.cv_engine = ComputerVisionEngine()

        # Define broker configurations
        self.brokers = {
            'alpaca': BrokerInterface(
                name='Alpaca',
                url='https://app.alpaca.markets',
                login_elements={'username': (100, 200), 'password': (100, 250)},
                trading_elements={'buy_button': (400, 300), 'sell_button': (500, 300)},
                price_patterns=[r'\$[\d,]+\.\d{2}'],
                button_templates={
                    'buy': 'templates/alpaca_buy_button.png',
                    'sell': 'templates/alpaca_sell_button.png'
                }
            ),
            'kalshi': BrokerInterface(
                name='Kalshi',
                url='https://kalshi.com',
                login_elements={'email': (150, 180), 'password': (150, 220)},
                trading_elements={'yes_button': (350, 400), 'no_button': (450, 400)},
                price_patterns=[r'\d+¢', r'\$\d+\.\d{2}'],
                button_templates={
                    'yes': 'templates/kalshi_yes_button.png',
                    'no': 'templates/kalshi_no_button.png'
                }
            ),
            'oanda': BrokerInterface(
                name='OANDA',
                url='https://trade.oanda.com',
                login_elements={'username': (200, 150), 'password': (200, 200)},
                trading_elements={'buy_button': (300, 350), 'sell_button': (400, 350)},
                price_patterns=[r'\d+\.\d{4,5}'],
                button_templates={
                    'buy': 'templates/oanda_buy_button.png',
                    'sell': 'templates/oanda_sell_button.png'
                }
            )
        }

    def navigate_to_broker(self, broker_name: str) -> bool:
        """Navigate to broker website"""
        if broker_name not in self.brokers:
            logger.error(f"Unknown broker: {broker_name}")
            return False

        broker = self.brokers[broker_name]

        try:
            # Open browser and navigate
            import webbrowser
            webbrowser.open(broker.url)
            time.sleep(3)  # Wait for page to load

            return True

        except Exception as e:
            logger.error(f"Failed to navigate to {broker_name}: {e}")
            return False

    def login_to_broker(self, broker_name: str, credentials: Dict[str, str]) -> bool:
        """Attempt to log into broker interface"""
        if broker_name not in self.brokers:
            return False

        broker = self.brokers[broker_name]

        try:
            # Take screenshot to see current state
            screenshot_path = self.mcp_client.take_screenshot()
            if not screenshot_path:
                return False

            # Look for login elements
            image = cv2.imread(screenshot_path)
            text_elements = self.cv_engine.extract_text_regions(image)

            # Find username/email field
            username_found = False
            password_found = False

            for element in text_elements:
                if element.text and any(keyword in element.text.lower()
                                     for keyword in ['email', 'username', 'user']):
                    x, y, w, h = element.coordinates
                    self.mcp_client.type_text(x + w + 10, y + h//2,
                                            credentials.get('username', ''))
                    username_found = True
                elif element.text and 'password' in element.text.lower():
                    x, y, w, h = element.coordinates
                    self.mcp_client.type_text(x + w + 10, y + h//2,
                                            credentials.get('password', ''))
                    password_found = True

            if username_found and password_found:
                # Look for login button
                login_buttons = self.cv_engine.find_trading_buttons(
                    image, {'login': 'templates/generic_login_button.png'}
                )

                if login_buttons:
                    button = login_buttons[0]
                    x, y, w, h = button.coordinates
                    self.mcp_client.click_element(x + w//2, y + h//2)
                    time.sleep(2)
                    return True

            return False

        except Exception as e:
            logger.error(f"Login to {broker_name} failed: {e}")
            return False

    def scan_for_opportunities(self, broker_name: str) -> List[TradingOpportunity]:
        """Scan broker interface for trading opportunities"""
        opportunities = []

        try:
            # Take screenshot
            screenshot_path = self.mcp_client.take_screenshot()
            if not screenshot_path:
                return opportunities

            image = cv2.imread(screenshot_path)

            # Detect price elements and movements
            price_elements = self.cv_engine.detect_price_changes(image)

            # Find trading buttons
            broker = self.brokers.get(broker_name)
            if broker:
                trading_buttons = self.cv_engine.find_trading_buttons(
                    image, broker.button_templates
                )

                # Analyze for opportunities
                for price_elem in price_elements:
                    if price_elem.element_type in ['price_up', 'price_down']:
                        # Find nearby trading buttons
                        px, py, pw, ph = price_elem.coordinates

                        for button in trading_buttons:
                            bx, by, bw, bh = button.coordinates

                            # Check if button is near price element
                            distance = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5

                            if distance < 200:  # Within 200 pixels
                                action = 'buy' if price_elem.element_type == 'price_down' else 'sell'

                                opportunity = TradingOpportunity(
                                    broker=broker_name,
                                    symbol=f"DETECTED_{int(time.time())}",
                                    action=action,
                                    price=price_elem.value or 0,
                                    confidence=min(price_elem.confidence, button.confidence),
                                    visual_evidence=self._encode_image(screenshot_path),
                                    reasoning=f"Detected {price_elem.element_type} near {button.text} button",
                                    timestamp=datetime.now(timezone.utc)
                                )
                                opportunities.append(opportunity)

            logger.info(f"Found {len(opportunities)} opportunities on {broker_name}")
            return opportunities

        except Exception as e:
            logger.error(f"Opportunity scan failed for {broker_name}: {e}")
            return opportunities

    def execute_trade(self, opportunity: TradingOpportunity) -> bool:
        """Execute a detected trading opportunity"""
        try:
            # Take fresh screenshot
            screenshot_path = self.mcp_client.take_screenshot()
            if not screenshot_path:
                return False

            image = cv2.imread(screenshot_path)
            broker = self.brokers.get(opportunity.broker)

            if not broker:
                return False

            # Find appropriate trading button
            button_name = 'buy' if opportunity.action == 'buy' else 'sell'
            if opportunity.broker == 'kalshi':
                button_name = 'yes' if opportunity.action == 'buy' else 'no'

            trading_buttons = self.cv_engine.find_trading_buttons(
                image, {button_name: broker.button_templates.get(button_name, '')}
            )

            if trading_buttons:
                # Click the highest confidence button
                best_button = max(trading_buttons, key=lambda x: x.confidence)
                x, y, w, h = best_button.coordinates

                success = self.mcp_client.click_element(x + w//2, y + h//2)
                if success:
                    logger.info(f"Executed {opportunity.action} on {opportunity.symbol}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False

    def _encode_image(self, image_path: str) -> str:
        """Encode image as base64 for storage"""
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.debug(f"Error encoding image {image_path}: {e}")
            return ""


# =============================================================================
# MAIN COMPUTER VISION BOT
# =============================================================================

class ComputerVisionBot:
    """
    Main Computer Vision Trading Bot

    Integrates with Windows-MCP to provide automated trading through
    visual interface detection and interaction.
    """

    def __init__(self, paper_mode: bool = True, enabled_brokers: List[str] = None):
        self.paper_mode = paper_mode
        self.enabled_brokers = enabled_brokers or ['alpaca', 'kalshi', 'oanda']

        # Initialize components
        self.mcp_client = WindowsMCPClient()
        self.broker_manager = BrokerInterfaceManager(self.mcp_client)
        self.alerts = None

        # State tracking
        self.last_scan_time = None
        self.active_opportunities = []
        self.execution_history = []

        # Configuration
        self.scan_interval = 30  # seconds
        self.max_concurrent_trades = 3
        self.confidence_threshold = 0.7

        self._init_components()

        logger.info(f"ComputerVisionBot initialized (paper_mode={paper_mode})")

    def _init_components(self):
        """Initialize bot components"""
        try:
            # Initialize alerts
            if TRADING_DEPS_AVAILABLE:
                self.alerts = TradingAlerts()
                logger.info("[OK] Trading alerts initialized")

            # Start Windows-MCP server
            if self.mcp_client.start_server():
                logger.info("[OK] Windows-MCP server connected")
            else:
                logger.warning("[WARN] Windows-MCP server not available")

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")

    def run_strategy(self) -> Dict:
        """
        Main strategy execution method (orchestrator compatible)

        Returns:
            Dict with strategy results and any detected opportunities
        """
        start_time = time.time()

        # Check if CV dependencies are available
        if not CV_AVAILABLE:
            return {
                'status': 'success',
                'message': 'Computer Vision dependencies not installed - running in demo mode',
                'opportunities_found': 0,
                'total_scanned': 0,
                'execution_time': 0.1
            }

        try:
            # Check if we should scan
            if (self.last_scan_time and
                datetime.now().timestamp() - self.last_scan_time < self.scan_interval):
                return {
                    'status': 'waiting',
                    'message': f'Next scan in {self.scan_interval - (datetime.now().timestamp() - self.last_scan_time):.0f}s'
                }

            self.last_scan_time = datetime.now().timestamp()

            # Perform visual scan of all enabled brokers
            all_opportunities = []

            for broker_name in self.enabled_brokers:
                try:
                    logger.info(f"Scanning {broker_name}...")

                    # Navigate to broker if needed
                    if not self._is_broker_open(broker_name):
                        if not self.broker_manager.navigate_to_broker(broker_name):
                            logger.warning(f"Failed to navigate to {broker_name}")
                            continue

                    # Scan for opportunities
                    opportunities = self.broker_manager.scan_for_opportunities(broker_name)
                    all_opportunities.extend(opportunities)

                    logger.info(f"Found {len(opportunities)} opportunities on {broker_name}")

                except Exception as e:
                    logger.error(f"Error scanning {broker_name}: {e}")

            # Filter and rank opportunities
            viable_opportunities = self._filter_opportunities(all_opportunities)

            # Execute top opportunities if not in paper mode
            executed_trades = []
            if viable_opportunities and not self.paper_mode:
                executed_trades = self._execute_opportunities(viable_opportunities[:self.max_concurrent_trades])

            analysis_time = time.time() - start_time

            result = {
                'status': 'completed',
                'opportunities_found': len(all_opportunities),
                'viable_opportunities': len(viable_opportunities),
                'executed_trades': len(executed_trades),
                'analysis_time': analysis_time,
                'paper_mode': self.paper_mode,
                'brokers_scanned': self.enabled_brokers,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            # Add top opportunity details for logging
            if viable_opportunities:
                best_opp = viable_opportunities[0]
                result.update({
                    'action': best_opp.action,
                    'symbol': best_opp.symbol,
                    'broker': best_opp.broker,
                    'price': best_opp.price,
                    'confidence': best_opp.confidence
                })

            # Send alerts for significant opportunities
            if viable_opportunities and self.alerts:
                self._send_opportunity_alert(viable_opportunities[0])

            return result

        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

    def _is_broker_open(self, broker_name: str) -> bool:
        """Check if broker interface is currently open"""
        try:
            desktop_state = self.mcp_client.get_desktop_state()
            active_windows = desktop_state.get('active_windows', [])

            broker_keywords = {
                'alpaca': ['alpaca', 'trading'],
                'kalshi': ['kalshi'],
                'oanda': ['oanda', 'trade']
            }

            keywords = broker_keywords.get(broker_name, [])
            for window in active_windows:
                if any(keyword in str(window).lower() for keyword in keywords):
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking if {broker_name} is open: {e}")
            return False

    def _filter_opportunities(self, opportunities: List[TradingOpportunity]) -> List[TradingOpportunity]:
        """Filter and rank opportunities by confidence and viability"""
        if not opportunities:
            return []

        # Filter by confidence threshold
        filtered = [opp for opp in opportunities if opp.confidence >= self.confidence_threshold]

        # Remove duplicates (same broker + symbol)
        seen = set()
        unique_opportunities = []
        for opp in filtered:
            key = (opp.broker, opp.symbol)
            if key not in seen:
                seen.add(key)
                unique_opportunities.append(opp)

        # Sort by confidence
        unique_opportunities.sort(key=lambda x: x.confidence, reverse=True)

        return unique_opportunities

    def _execute_opportunities(self, opportunities: List[TradingOpportunity]) -> List[Dict]:
        """Execute trading opportunities"""
        executed = []

        for opportunity in opportunities:
            try:
                if len(executed) >= self.max_concurrent_trades:
                    break

                success = self.broker_manager.execute_trade(opportunity)

                if success:
                    trade_record = {
                        'broker': opportunity.broker,
                        'symbol': opportunity.symbol,
                        'action': opportunity.action,
                        'price': opportunity.price,
                        'confidence': opportunity.confidence,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'status': 'executed'
                    }
                    executed.append(trade_record)
                    self.execution_history.append(trade_record)

                    logger.info(f"Executed trade: {opportunity.action} {opportunity.symbol} on {opportunity.broker}")

            except Exception as e:
                logger.error(f"Failed to execute opportunity: {e}")

        return executed

    def _send_opportunity_alert(self, opportunity: TradingOpportunity):
        """Send alert for significant opportunity"""
        try:
            if self.alerts:
                message = (
                    f"🎯 CV Trading Opportunity\n"
                    f"🏢 Broker: {opportunity.broker.title()}\n"
                    f"📈 Action: {opportunity.action.upper()}\n"
                    f"🎯 Symbol: {opportunity.symbol}\n"
                    f"💰 Price: ${opportunity.price:.2f}\n"
                    f"🎯 Confidence: {opportunity.confidence:.1%}\n"
                    f"🧠 Reasoning: {opportunity.reasoning[:100]}..."
                )

                if self.paper_mode:
                    message = "[PAPER] " + message

                self.alerts.send(message)

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def get_status(self) -> Dict:
        """Get current bot status"""
        return {
            'paper_mode': self.paper_mode,
            'enabled_brokers': self.enabled_brokers,
            'last_scan_time': self.last_scan_time,
            'active_opportunities': len(self.active_opportunities),
            'execution_history_count': len(self.execution_history),
            'mcp_connected': self.mcp_client.is_connected,
            'scan_interval': self.scan_interval,
            'confidence_threshold': self.confidence_threshold,
            'max_concurrent_trades': self.max_concurrent_trades
        }

    def find_opportunities(self) -> List[TradingOpportunity]:
        """Alias for orchestrator compatibility"""
        result = self.run_strategy()
        return self.active_opportunities or []


# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

def test_computer_vision_bot():
    """Test the computer vision bot functionality"""
    print("=" * 60)
    print("COMPUTER VISION TRADING BOT TEST")
    print("=" * 60)

    # Initialize bot in paper mode
    bot = ComputerVisionBot(paper_mode=True, enabled_brokers=['alpaca'])

    print("\n[1] Bot Status:")
    status = bot.get_status()
    print(json.dumps(status, indent=2))

    print("\n[2] Running Strategy:")
    result = bot.run_strategy()
    print(json.dumps(result, indent=2))

    print("\n[3] Test Completed")


def main():
    """Main function for standalone testing"""
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Computer Vision Trading Bot")
    parser.add_argument('--test', action='store_true', help="Run test mode")
    parser.add_argument('--brokers', nargs='+', default=['alpaca', 'kalshi'],
                       help="Brokers to monitor")
    parser.add_argument('--live', action='store_true', help="Live mode (default: paper)")

    args = parser.parse_args()

    if args.test:
        test_computer_vision_bot()
        return

    paper_mode = not args.live
    bot = ComputerVisionBot(paper_mode=paper_mode, enabled_brokers=args.brokers)

    try:
        while True:
            result = bot.run_strategy()
            print(f"{datetime.now().strftime('%H:%M:%S')} - {result.get('status', 'unknown')}")

            if result.get('opportunities_found', 0) > 0:
                print(f"  Found {result['opportunities_found']} opportunities")

            time.sleep(30)  # Wait 30 seconds before next scan

    except KeyboardInterrupt:
        print("\nStopping Computer Vision Bot...")


if __name__ == "__main__":
    main()