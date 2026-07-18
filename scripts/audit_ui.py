"""Headless browser regression. Run after starting uvicorn; no paid LLM calls."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8017')
    args = parser.parse_args()
    output = ROOT / 'reports' / 'ui_audit'
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000}, accept_downloads=True)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.locator('#engine').select_option('baseline')
        expect(page.locator('#search_mode')).to_be_disabled()
        page.locator('#evaluate').click()
        expect(page.locator('#status')).to_contain_text('Готово', timeout=60000)
        expected_result = json.loads(page.locator('#output').text_content())
        with page.expect_download() as event:
            page.locator('#download').click()
        event.value.save_as(output / 'downloaded_result.json')
        assert json.loads((output / 'downloaded_result.json').read_text(encoding='utf-8')) == expected_result
        page.locator('#query').fill('   ')
        expect(page.locator('#download')).to_be_disabled()
        page.locator('#evaluate').click()
        assert page.locator('#query').evaluate('(el) => !el.validity.valid')
        page.locator('#negative').click()
        expect(page.locator('#query')).to_have_value('кальянная для мероприятий')
        expect(page.locator('#output')).to_have_text('Нет данных')
        page.locator('#engine').select_option('llm')
        expect(page.locator('#search_mode')).to_be_enabled()
        page.locator('#search_mode').select_option('always')
        # A controlled failure checks rendering without spending API credits.
        page.route('**/score', lambda route: route.fulfill(status=502, json={'detail': 'Тестовый отказ API'}))
        page.locator('#evaluate').click()
        expect(page.locator('#status')).to_contain_text('Тестовый отказ API')
        expect(page.locator('#evaluate')).to_be_enabled()
        expect(page.locator('#negative')).to_be_enabled()
        expect(page.locator('#download')).to_be_disabled()
        page.unroute('**/score')
        page.locator('#engine').select_option('baseline')
        page.locator('#evaluate').click()
        expect(page.locator('#status')).to_contain_text('Готово', timeout=60000)
        for width in [1440, 390, 320]:
            page.set_viewport_size({'width': width, 'height': 1000})
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
            for name in ['query', 'organization_name', 'category', 'address', 'prices_summarized', 'review_snippets', 'engine', 'search_mode']:
                box = page.locator('#' + name).bounding_box()
                assert box and box['width'] > 0 and box['x'] >= 0 and box['x'] + box['width'] <= width
            page.screenshot(path=output / f'width_{width}.png', full_page=True)
        assert not errors, errors
        browser.close()
    print('PASS: real baseline, download contents, stale state, whitespace validation, modes, controlled API error, desktop/mobile layout, no JS errors')


if __name__ == '__main__':
    main()
