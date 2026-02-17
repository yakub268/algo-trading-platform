#!/usr/bin/env python3
"""
PWA Icon Generator
==================

Generates PWA icons in various sizes from a base design.
For production, replace this with actual icon files.

Author: Trading Bot System
Created: February 2026
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_trading_icon(size: int, output_path: str):
    """Create a simple trading icon for PWA"""
    # Create a new image with transparent background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background circle
    margin = size // 10
    draw.ellipse([margin, margin, size - margin, size - margin],
                fill=(26, 26, 46, 255), outline=(0, 102, 204, 255), width=max(2, size // 50))

    # Simple chart icon
    chart_margin = size // 4
    chart_width = size - (chart_margin * 2)
    chart_height = size // 3

    # Chart bars
    bar_width = chart_width // 5
    for i in range(4):
        bar_height = (chart_height * (i + 2)) // 5
        x = chart_margin + (i * chart_width // 4)
        y = size - chart_margin - bar_height

        draw.rectangle([x, y, x + bar_width, size - chart_margin],
                      fill=(0, 212, 170, 255))

    # Add $ symbol
    try:
        font_size = max(12, size // 8)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        text = "$"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = (size - text_width) // 2
        text_y = chart_margin

        draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)
    except Exception as e:
        print(f"Warning: Could not add text to icon: {e}")

    # Save the icon
    img.save(output_path, 'PNG')
    print(f"Generated icon: {output_path} ({size}x{size})")

def generate_all_icons():
    """Generate all required PWA icon sizes"""
    icon_sizes = [72, 96, 128, 144, 152, 192, 384, 512]

    # Create icons directory if it doesn't exist
    icons_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(icons_dir, exist_ok=True)

    for size in icon_sizes:
        output_path = os.path.join(icons_dir, f'icon-{size}x{size}.png')
        create_trading_icon(size, output_path)

    # Create favicon
    create_trading_icon(32, os.path.join(icons_dir, 'icon-32x32.png'))
    create_trading_icon(16, os.path.join(icons_dir, 'icon-16x16.png'))

    print("\n✅ All PWA icons generated successfully!")
    print("\n📝 Note: For production use, replace these with professionally designed icons.")
    print("   Consider hiring a designer or using tools like:")
    print("   - Figma")
    print("   - Adobe Illustrator")
    print("   - Canva")
    print("   - Icon generator websites")

if __name__ == '__main__':
    try:
        generate_all_icons()
    except ImportError:
        print("❌ PIL (Pillow) not found. Install it with:")
        print("   pip install Pillow")
        print("\nAlternatively, you can:")
        print("1. Create icons manually using image editing software")
        print("2. Use online PWA icon generators")
        print("3. Hire a designer for professional icons")
    except Exception as e:
        print(f"❌ Error generating icons: {e}")
        print("\nPlease create the following icon files manually:")
        for size in [72, 96, 128, 144, 152, 192, 384, 512]:
            print(f"   - icon-{size}x{size}.png")