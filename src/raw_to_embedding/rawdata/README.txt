SSL 原始数据（rawdata）说明
========================

本目录由 scripts/build_rawdata_and_audit.py 从 data/ssl_crawl/ 整理生成，并附带 URL 可达性审计。

目录结构
--------
urls/ingest/urls.txt   合并去重后的 24 个页面 URL（推荐：python main.py --url-dir ./rawdata/urls/ingest）
urls/           分类/清单（多文件；勿对整个 urls/ 做 --url-dir，会重复加载）
pdfs/           已下载 PDF 副本
  www_umb_media/    主站 media 上的 4 个 PDF（与 pdf_links.json 一致）
  scholarworks/     ScholarWorks 系列 15 个 article*.pdf（viewcontent）
audit/            audit_report.json —— 对 24 个 HTML 页、19 条 PDF URL 的 HTTP 检查结果

查漏补缺结论（最近一次审计）
----------------------------
- 24 个 HTML 页面：全部为 200。
- ScholarWorks 首页解析：保存的 index 与线上均为 15 条 viewcontent 链接，与 scholarworks_viewcontent_pdf_urls.txt 一致，无遗漏。
- viewcontent 链接用 requests 测试常为 202（AWS WAF）；属预期，不影响已落盘的 PDF 文件。

重新生成
--------
在 `src/raw_to_embedding` 目录执行：
  python scripts/build_rawdata_and_audit.py

喂给 pipeline 的示例
-------------------
  python -m raw_to_embedding.pipeline --url-dir ./src/raw_to_embedding/rawdata/urls/ingest ^
    --pdf ./rawdata/pdfs/www_umb_media/Who-Counts-In-Climate-Resilience-Web.pdf ...
