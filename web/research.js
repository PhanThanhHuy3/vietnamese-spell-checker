function button(label, action) {
  const element = node("button", label);
  element.type = "button";
  element.className = "secondary";
  element.onclick = action;
  return element;
}

function link(label, href) {
  const element = node("a", label);
  element.href = href;
  return element;
}

function option(label, value) {
  return Object.assign(node("option", label), {value});
}

function card(title, ...children) {
  const element = node("article", "");
  element.append(node("h2", title), ...children);
  return element;
}

function tableRow(values) {
  const row = node("tr", "");
  row.append(...values.map((value) => node("td", value)));
  return row;
}

function pretty(data) {
  return node("pre", JSON.stringify(data, null, 2));
}

function download(name, content, type = "application/json") {
  const url = URL.createObjectURL(new Blob([content], {type}));
  const anchor = link("", url);
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function showError(id, error) {
  byId(id).textContent = error.message;
}

const formatNumber = new Intl.NumberFormat("vi-VN");
let modelState = {};
const modelLabels = {
  ready: "Sẵn sàng inference", validated: "Đang chờ kiểm tra inference",
  untrained: "Chưa huấn luyện", unverified: "Checkpoint chưa xác minh", incompatible: "Không tương thích",
};

function showModel(info, key, probe) {
  const details = node("div", "");
  if (probe) {
    details.append(node("p", probe.reason), ...(probe.diagnostic ? [pretty(probe.diagnostic)] : []));
  }
  const result = card(info.name, node("p", info.role + " · " + modelLabels[info.state]), node("p", info.reason));
  const actions = node("div", "");
  actions.className = "controls";
  actions.append(button("Xem kiến trúc", () => details.replaceChildren(node("p", info.architecture),
    pretty({framework: info.framework, tokenizer: info.tokenizer, nhiệm_vụ: info.objective, dữ_liệu: info.dataset}))));
  if (info.url) {
    const source = link("Xem nguồn pretrained", info.url);
    source.target = "_blank";
    source.rel = "noreferrer";
    actions.append(source);
  }
  if (key !== "baseline") {
    actions.append(button("Xem checkpoint", async () => {
      try {
        const value = await request("/api/models/" + key);
        details.replaceChildren(pretty({thư_mục: value.checkpoint_path, kích_thước: value.weights_bytes,
          tệp: value.files, metadata: value.metadata, trạng_thái: value.reason}));
      } catch (error) { details.textContent = error.message; }
    }), button("Kiểm tra checkpoint", async (event) => {
      const control = event.currentTarget;
      control.disabled = true;
      details.textContent = "Đang kiểm tra checkpoint…";
      try {
        const value = await request("/api/models/" + key + "?probe=1");
        probedCheckpoints.set(key, value);
        details.replaceChildren(node("p", value.reason), ...(value.diagnostic ? [pretty(value.diagnostic)] : []));
        await loadModels();
      } catch (error) { details.textContent = error.message; }
      finally { control.disabled = false; }
    }));
    const configuration = button("Xem cấu hình huấn luyện", () => details.replaceChildren(pretty(info.metadata.training)));
    const history = button("Xem quá trình huấn luyện", () => details.replaceChildren(lossGraph(info.metadata.history || [])));
    configuration.disabled = history.disabled = !info.metadata;
    actions.append(configuration, history);
  }
  actions.append(button("Xem kết quả thực nghiệm", async () => {
    navigate("report");
    await loadReports();
    byId("report-model").value = key;
    renderReports();
  }));
  const tryModel = button("Thử mô hình", () => {
    tryModel(key);
  });
  tryModel.disabled = !info.available;
  actions.append(tryModel);
  if (info.metadata) {
    const meta = info.metadata;
    const last = (meta.history || []).filter((entry) => Number.isFinite(entry.loss)).at(-1);
    const parameters = [
      ["Kiến trúc", info.architecture], ["Framework", info.framework], ["Dataset", info.dataset],
      ["Phương pháp", "Fine-tuning"], ["Tham số", formatNumber.format(meta.parameters)],
      ["Epoch", meta.epochs_completed], ["Training loss gần nhất", last?.loss],
      ["Thiết bị huấn luyện", meta.device], ["Thời gian huấn luyện", Math.round(meta.train_runtime_s) + " giây"],
      ["Phiên bản pretrained", meta.pretrained_revision], ["SHA-256 dữ liệu", meta.raw_sha256],
      ["Checkpoint", meta.training_id], ["Correction F1 Validation", f1(meta.calibration?.selected?.correction)]
    ];
    const list = node("dl", "");
    for (const [label, value] of parameters) if (value != null) list.append(node("dt", label), node("dd", value));
    result.append(list);
  }
  result.append(actions, details);
  return result;
}

let modelLoadRevision = 0;
const probedCheckpoints = new Map();
async function loadModels() {
  const revision = ++modelLoadRevision;
  const {models, recommendation, max_text_length: maxTextLength} = await request("/api/models");
  if (revision !== modelLoadRevision) return;
  byId("model-status").textContent = "";
  if (JSON.stringify(models) === JSON.stringify(modelState)) return;
  maximumText = maxTextLength;
  for (const [key, checkpoint] of probedCheckpoints) {
    const current = models[key];
    if (checkpoint.state !== current?.state || checkpoint.reason !== current?.reason ||
        checkpoint.metadata?.training_id !== current?.metadata?.training_id) {
      probedCheckpoints.delete(key);
    }
  }
  const previousModels = modelState;
  modelState = models;
  const selector = byId("check-model");
  const selected = selector.value;
  const known = Object.entries(models);
  selector.replaceChildren(option("Tự động", "auto"), ...known.map(([key, info]) => Object.assign(
    option(info.available ? info.name : info.name + " · " + modelLabels[info.state], key),
    {disabled: !info.available},
  )));
  selector.value = models[selected]?.available ? selected : "auto";
  if (selected && (selector.value !== selected ||
      previousModels[selected]?.metadata?.training_id !== models[selected]?.metadata?.training_id)) invalidateCheck();
  selector.onchange = () => {
    const current = models[selector.value];
    const selectedName = current ? "Đang chọn " + current.name + "." : "Tự động ưu tiên " + names[recommendation.model] + ".";
    const unavailable = models.byt5?.available === false ? " ByT5-small chưa có checkpoint sẵn sàng." : "";
    byId("check-model-info").textContent = selectedName + unavailable;
  };
  selector.onchange();
  byId("home-models").replaceChildren(...known.map(([, info]) => card(info.name, node("p", modelLabels[info.state]))));
  byId("model-cards").replaceChildren(...known.map(([key, info]) => showModel(info, key, probedCheckpoints.get(key))));
  const previousChoices = new Map([...byId("compare-models").querySelectorAll("input:not(:disabled)")]
    .map((box) => [box.value, box.checked]));
  byId("compare-models").replaceChildren(...known.map(([key, info]) => {
    const label = node("label", "");
    const box = Object.assign(node("input", ""), {type: "checkbox", name: "model", value: key,
      defaultChecked: info.available, checked: info.available && (previousChoices.get(key) ?? true),
      disabled: !info.available});
    label.append(box, document.createTextNode(" " + info.name + (info.available ? "" : " · " + modelLabels[info.state])));
    return label;
  }));
  const active = known.filter(([, info]) => info.available).map(([key]) => key);
  byId("compare-submit").textContent = active.length === 1 ? "Chạy thử baseline" : "So sánh";
  if (active.length !== previousChoices.size || active.some((key) => !previousChoices.has(key) ||
      previousModels[key]?.metadata?.training_id !== models[key].metadata?.training_id)) clearComparison();
}

const compareSamples = {missing: "ban la ai vay", regional: "Hôm nai tôi đến chường để học lập chình.",
  context: "Tôi đang sữa bài.", clean: "Tôi uống sữa."};
let compareRevision = 0;
function clearComparison() {
  compareRevision++;
  byId("comparison").replaceChildren();
  byId("compare-status").textContent = Object.values(modelState).filter((model) => model.available).length < 2
    ? "Cần ít nhất hai mô hình sẵn sàng để so sánh." : "";
}
byId("compare-form").addEventListener("reset", clearComparison);
byId("compare-form").addEventListener("input", clearComparison);
byId("compare-sample").onchange = (event) => {
  if (compareSamples[event.target.value]) byId("compare-text").value = compareSamples[event.target.value];
  clearComparison();
};
onForm("compare-form", "compare-status", async () => {
  const models = [...byId("compare-models").querySelectorAll("input:checked:not(:disabled)")].map((box) => box.value);
  if (!models.length) throw new Error("Chọn mô hình để chạy.");
  const version = ++compareRevision;
  const reference = byId("compare-reference").value;
  const data = await request("/api/compare", {text: checkedInput(byId("compare-text").value), models,
    ...(reference.trim() ? {reference: checkedInput(reference)} : {})});
  if (version !== compareRevision) return;
  byId("comparison").replaceChildren(...Object.entries(data.results).map(([key, result]) => {
    if (result.error) return card(names[key], node("p", result.error));
    const output = node("div", "");
    output.className = "text-output";
    annotate(result.original_text, result.errors, new Map(result.errors.map((edit) => [edit.id, edit.suggestion])), output, false);
    const status = node("p", "");
    const details = node("details", "");
    const list = node("ul", "");
    list.append(...result.errors.map((error) => node("li", error.original + " → " + error.suggestion + " · Vị trí " + error.position)));
    details.append(node("summary", "Xem chi tiết"), result.errors.length ? list : node("p", "Không phát hiện lỗi."));
    const metrics = result.reference_metrics;
    if (metrics) details.append(pretty(metrics));
    return card(names[key], output, node("p", result.suggestion_count + " gợi ý · Xử lý " + result.latency_ms + " ms"),
      details, button("Sao chép kết quả", () => copyText(result.predicted_text, status)), status, link("Xem thông tin mô hình", "/lab/models"));
  }));
  byId("compare-status").textContent = data.message || "Đã chạy trên cùng văn bản đầu vào.";
});

function barChart(title, entries, scale) {
  const box = node("div", "");
  box.className = "chart";
  const max = scale || Math.max(1, ...entries.map(([, value]) => value));
  for (const [label, value] of entries) {
    const meter = Object.assign(node("meter", ""), {max, value});
    meter.setAttribute("aria-label", label);
    const row = node("div", "");
    row.append(node("span", label), meter, node("span", formatNumber.format(value)));
    box.append(row);
  }
  return card(title, box);
}

async function loadDataset() {
  const data = await request("/api/dataset/stats");
  const splits = Object.entries(data.splits);
  const total = (key) => splits.reduce((count, [, row]) => count + row[key], 0);
  byId("home-stats").replaceChildren(card(formatNumber.format(total("samples")), node("p", "Câu dữ liệu")),
    card(formatNumber.format(total("annotated_errors")), node("p", "Lỗi gán nhãn")), card("Transformer", node("p", "Kiến trúc ViT5")));
  byId("dataset-summary").textContent = data.source.authors + " · Seed " + data.seed + " · " + data.duplicates
    + " cặp trùng đã loại · " + data.rejected_lines + " dòng không hợp lệ · " + data.source.terms;
  byId("dataset-table").replaceChildren(...splits.map(([key, row]) => tableRow([
    {train: "Train", val: "Validation", test: "Test"}[key], row.samples, row.annotated_errors, row.clean_sentences,
    row.syllables, row.mean_length.toFixed(1)
  ])));
  const aggregate = (key) => {
    const result = {};
    splits.forEach(([, row]) => Object.entries(row[key]).forEach(([group, count]) => { result[group] = (result[group] || 0) + count; }));
    return Object.entries(result);
  };
  const lengths = {};
  aggregate("length_histogram").forEach(([length, count]) => {
    const low = Math.floor(Number(length) / 10) * 10;
    const name = low + "–" + (low + 9);
    lengths[name] = (lengths[name] || 0) + count;
  });
  byId("distribution").replaceChildren(...barChart("Số lỗi trong mỗi câu", aggregate("errors_per_sentence").map(([key, value]) => [key + " lỗi", value])).children);
  const types = aggregate("error_types");
  byId("dataset-charts").replaceChildren(barChart("Độ dài câu theo âm tiết", Object.entries(lengths)),
    barChart("Nhóm lỗi theo annotation", types), barChart("Mẫu Train tăng cường đã lưu", Object.entries(data.augmentation)));
  const tokens = button("Xem phân bố token ViT5", async () => {
    tokens.disabled = true;
    try {
      const {token_histograms: tokenHistograms} = await request("/api/dataset/stats?tokens=1");
      const bins = {};
      Object.values(tokenHistograms).forEach((histogram) => Object.entries(histogram).forEach(([length, count]) => {
        const low = Math.floor(Number(length) / 10) * 10;
        const label = low + "–" + (low + 9);
        bins[label] = (bins[label] || 0) + count;
      }));
      if (Object.keys(bins).length) tokens.replaceWith(barChart("Độ dài theo token ViT5", Object.entries(bins)));
      else tokens.replaceWith(node("p", "Chưa có tokenizer ViT5 cục bộ."));
    } catch (error) {
      showError("dataset-summary", error);
      tokens.disabled = false;
    }
  });
  byId("token-distribution").replaceChildren(tokens);
  byId("data-kind").append(...types.map(([key]) => option(key, key)));
  byId("export-data").onclick = () => download("vsec-statistics.json", JSON.stringify(data, null, 2));
}

byId("verify-data").onclick = async () => {
  const result = byId("integrity");
  result.hidden = false;
  result.textContent = "Đang đối chiếu manifest và nội dung…";
  try { result.textContent = JSON.stringify(await request("/api/dataset/integrity"), null, 2); }
  catch (error) { result.textContent = error.message; }
};

async function sampleDetail(split, id, withTokens = false) {
  try {
    const result = await request("/api/dataset/samples?" + new URLSearchParams({split, id, tokenize: withTokens ? "1" : "0"}));
    const sample = result.samples[0];
    if (!sample) return;
    const detail = byId("sample-detail");
    detail.replaceChildren(node("h3", sample.id), node("p", "Input · " + sample.input_text), node("p", "Target · " + sample.target_text));
    const content = node("div", "");
    detail.append(button("Xem annotation", () => content.replaceChildren(pretty({
      trước_chuẩn_hóa: sample.raw_text, sau_chuẩn_hóa: sample.normalized_text, annotation: sample.annotation
    }))), button("Xem tokenization", () => sampleDetail(split, id, true)), content);
    if (withTokens) content.append(pretty(sample.tokenization));
  } catch (error) { showError("data-sample-status", error); }
}

let samplePage = 0;
let sampleQuery = null;
let sampleRevision = 0;
async function renderSamples({random = false, newQuery = false} = {}) {
  if (newQuery) {
    samplePage = 0;
    sampleQuery = {split: byId("data-split").value, q: byId("data-search").value, type: byId("data-kind").value};
  }
  const revision = ++sampleRevision;
  const result = await request("/api/dataset/samples?" + new URLSearchParams({...sampleQuery,
    offset: String(samplePage * 12), limit: "12", random: random ? "1" : "0"}));
  if (revision !== sampleRevision) return;
  byId("sample-detail").replaceChildren();
  byId("sample-list").replaceChildren(...result.samples.map((sample) => {
    const row = tableRow(["", sample.input_text, sample.target_text, sample.types.join(", ") || "—"]);
    row.firstChild.append(button(sample.id, () => sampleDetail(sampleQuery.split, sample.id)));
    return row;
  }));
  byId("samples-prev").disabled = random || samplePage === 0;
  byId("samples-next").disabled = random || result.offset + result.samples.length >= result.matched;
  byId("data-sample-status").textContent = random ? "Mẫu ngẫu nhiên" : result.matched + " mẫu phù hợp · Trang " + (samplePage + 1);
  if (random && result.samples.length) await sampleDetail(sampleQuery.split, result.samples[0].id);
}
onForm("dataset-form", "data-sample-status", (submitter) => renderSamples({newQuery: true, random: submitter?.value === "random"}));
for (const [id, delta] of [["samples-prev", -1], ["samples-next", 1]]) {
  byId(id).onclick = async () => {
    const previous = samplePage;
    samplePage += delta;
    byId("samples-prev").disabled = byId("samples-next").disabled = true;
    try { await renderSamples(); }
    catch (error) {
      samplePage = previous;
      byId("samples-prev").disabled = samplePage === 0;
      byId("samples-next").disabled = false;
      showError("data-sample-status", error);
    }
  };
}

function showNoiseProbability() {
  byId("noise-value").textContent = Math.round(byId("noise-prob").value * 100) + "%";
}
byId("noise-prob").oninput = showNoiseProbability;
byId("noise-form").onreset = () => {
  byId("noise-result").replaceChildren();
  byId("noise-status").textContent = "";
  byId("export-noise").hidden = byId("copy-noise").hidden = true;
  setTimeout(showNoiseProbability, 0);
};
onForm("noise-form", "noise-status", async (submitter) => {
  let seed = Number(byId("noise-seed").value);
  if (submitter?.value === "again") seed = (seed + 1) % 4294967296;
  byId("noise-seed").value = seed;
  const report = await request("/api/augment", {text: checkedInput(byId("noise-text").value), type: byId("noise-type").value,
    count: Number(byId("noise-count").value), seed, noise_prob: Number(byId("noise-prob").value)});
  byId("noise-result").replaceChildren(...report.variants.map((result, index) => {
    const output = node("p", result.input_text);
    output.className = "text-output";
    const status = node("p", "");
    const list = node("ul", "");
    list.append(...result.differences.map((edit) => node("li", edit.original_segment + " → " + edit.corrected_segment + " · " + edit.category)));
    return card("Biến thể " + (index + 1), node("h3", "Câu chứa lỗi"), output, node("h3", "Câu chuẩn"), node("p", result.target_text), node("p", result.differences.length + " thay đổi"), list,
      button("Sao chép", () => copyText(result.input_text, status)),
      button("Chuyển sang kiểm tra", () => {
        input.value = result.input_text;
        input.dispatchEvent(new Event("input"));
        navigate("check");
      }), status);
  }));
  byId("noise-status").textContent = report.variants.length + " biến thể · Seed " + seed;
  byId("export-noise").hidden = byId("copy-noise").hidden = false;
  const jsonl = report.variants.map((row) => JSON.stringify(row)).join("\n") + "\n";
  byId("export-noise").onclick = () => download("vsec-augmentation.jsonl", jsonl);
  byId("copy-noise").onclick = () => copyText(jsonl, byId("noise-status"));
});

const noiseSplits = ["full_diacritics", "partial_diacritics", "mixed"];
const percent = (value) => typeof value === "number" ? (value * 100).toFixed(2) + "%" : "—";
const f1 = (value) => typeof value?.f1 === "number" ? value.f1.toFixed(2) + "%" : "—";
let savedRuns = [];

function lossGraph(history) {
  const entries = history.flatMap((entry, i) => [["Train", entry.loss], ["Validation", entry.eval_loss]]
    .filter(([, value]) => Number.isFinite(value)).map(([group, value]) => [group + " " + (entry.step ?? i), value]));
  return entries.length ? barChart("Training và validation loss", entries.slice(-50)) : node("p", "Lần chạy chưa có nhật ký loss.");
}

async function predictions(run, kind, container) {
  try {
    const result = await request("/api/experiments/" + encodeURIComponent(run.id) + "/predictions?error=" + encodeURIComponent(kind));
    const items = result.samples.map((row) => card(row.id, node("p", "Đầu vào · " + row.input_text),
      node("p", "Đáp án chuẩn · " + (row.matched_reference || row.target_text)), node("p", "Dự đoán · " + row.prediction),
      ...(row.raw_prediction ? [node("p", "Trước hậu xử lý · " + row.raw_prediction)] : []),
      node("p", row.errors ? "Bỏ sót " + row.errors.missed + " · Sửa quá mức " + row.errors.overcorrected
        + " · Thay sai " + row.errors.wrong_replacement : "Khớp đáp án")));
    container.replaceChildren(node("p", result.matched + " mẫu, hiển thị " + items.length), ...items);
  } catch (error) { container.textContent = error.message; }
}

function runDetails(run, metrics, sourceSplit) {
  const details = node("div", "");
  const select = node("select", "");
  select.setAttribute("aria-label", "Lọc ví dụ lỗi");
  select.append(option("Tất cả", ""), option("Bỏ sót", "missed"), option("Phát hiện nhầm", "overcorrected"),
    option("Đúng vị trí, sai từ", "wrong_replacement"), option("Tự thêm hoặc xóa", "structural"));
  select.onchange = () => predictions(run, select.value, details);
  return card(run.run, node("p", noiseSplits.includes(sourceSplit)
    ? "Mô phỏng từ câu chuẩn VSEC Test · Seed 42." : (run.training_id ? "Checkpoint " + run.training_id : "Baseline")),
    button("Xem chi tiết", () => details.replaceChildren(pretty(metrics))),
    button("Xem cấu hình", () => details.replaceChildren(pretty({training: run.training, generation: run.generation,
      mô_hình: run.model, tập: run.split, hash: run.split_sha256}))),
    button("Xem biểu đồ Loss", async () => {
      try { details.replaceChildren(lossGraph((await request("/api/experiments/" + run.id + "/history")).history)); }
      catch (error) { details.textContent = error.message; }
    }), ...(noiseSplits.includes(sourceSplit) ? [] : [select, button("Xem mẫu dự đoán", () => predictions(run, select.value, details))]),
    details);
}

function renderReports() {
  const split = byId("report-split").value;
  const model = byId("report-model").value;
  const runId = byId("report-run").value;
  const pool = savedRuns.filter((run) => (!model || run.model === model)
    && (noiseSplits.includes(split) ? run.split === "test" && run.metrics.noise_groups?.[split] : run.split === split));
  byId("report-run").replaceChildren(option("Tất cả", ""), ...pool.map((run) => option(run.run, run.id)));
  byId("report-run").value = pool.some((run) => run.id === runId) ? runId : "";
  const active = pool.filter((run) => !byId("report-run").value || run.id === byId("report-run").value);
  const report = active.map((run) => ({run, metrics: noiseSplits.includes(split) ? run.metrics.noise_groups[split] : run.metrics}));
  byId("report-status").textContent = report.length ? "" : "Chưa có kết quả đã xác minh cho lựa chọn này.";
  byId("report-table").replaceChildren(...report.map(({run, metrics: m}) => tableRow([run.run, names[run.model], m.num_samples,
    percent(m.exact_match), percent(m.token_accuracy), m.bleu.toFixed(2), percent(m.cer), percent(m.wer),
    f1(m.detection), f1(m.correction), percent(m.clean_reference_control?.rate)])));
  byId("report-overview").replaceChildren(...report.map(({run, metrics: m}) => card(names[run.model],
    node("p", run.run + " · " + formatNumber.format(m.num_samples) + " câu"),
    barChart("Hiệu quả phát hiện và sửa lỗi", [
      ["Detection F1", m.detection.f1], ["Correction F1", m.correction.f1],
      ["Exact Match", +(100 * m.exact_match).toFixed(2)], ["CER", +(100 * m.cer).toFixed(2)]
    ], 100))));
  byId("report-history").replaceChildren(...report.map(({run}) => card(names[run.model], lossGraph(run.history || []))));
  byId("report-errors").replaceChildren(...report.map(({run, metrics: m}) => card(names[run.model],
    node("p", "Bỏ sót vị trí: " + m.detection.false_negative + " · Phát hiện nhầm: " + m.detection.false_positive +
      " · Đúng vị trí, sai từ: " + (m.detection.true_positive - m.correction.true_positive)))));
  byId("report-predictions").replaceChildren(...(noiseSplits.includes(split) ? [] : report.map(({run, metrics}) => runDetails(run, metrics, split))));
  byId("report-details").replaceChildren(...report.map(({run, metrics}) => runDetails(run, metrics, split)));
  byId("report-export").disabled = !report.length;
  byId("report-export").onclick = () => download("vsec-evaluation.json", JSON.stringify({split, results: report}, null, 2));
}

async function loadReports() {
  const {runs} = await request("/api/experiments");
  savedRuns = runs;
  const selected = byId("report-model").value;
  byId("report-model").replaceChildren(option("Tất cả", ""),
    ...[...new Set(runs.map((run) => run.model))].map((key) => option(names[key] || key, key)));
  if ([...byId("report-model").options].some((item) => item.value === selected)) byId("report-model").value = selected;
  renderReports();
  const runsOnTest = runs.filter((run) => run.split === "test");
  const engines = Object.keys(names).filter((key) => modelState[key] || runsOnTest.some((run) => run.model === key) || key !== "identity");
  byId("compare-scores").replaceChildren(...engines.map((key) => {
    const matches = runsOnTest.filter((entry) => entry.model === key);
    return card(names[key], ...(matches.length ? matches.map((entry) => node("p", entry.run +
      " · Detection F1 " + f1(entry.metrics.detection) + " · Correction F1 " + f1(entry.metrics.correction)))
      : [node("p", "Chưa có kết quả VSEC Test đã xác minh.")]));
  }));
  const vit5 = runsOnTest.filter((run) => run.model === "vit5" && run.training_id)
    .sort((a, b) => (b.evaluated_at || "").localeCompare(a.evaluated_at || ""))[0];
  byId("lab-summary").textContent = vit5 ? "ViT5-base trên VSEC Test · Detection F1 " + f1(vit5.metrics.detection)
    + " · Correction F1 " + f1(vit5.metrics.correction) : "Chưa có kết quả ViT5 trên VSEC Test.";
}

byId("report-header").replaceChildren(...["Lần chạy", "Mô hình", "Số câu", "Exact Match", "Token Accuracy", "BLEU-4", "CER", "WER",
  "Detection F1", "Correction F1", "Thay đổi câu chuẩn"].map((label) => node("th", label)));
for (const id of ["report-split", "report-model", "report-run"]) byId(id).onchange = renderReports;
byId("report-refresh").onclick = () => loadReports().catch((error) => showError("report-status", error));
function refreshModels() {
  loadModels().catch((error) => { showError("model-status", error); showError("check-model-info", error); });
}
for (const section of sections) {
  const tabs = [...section.querySelectorAll("[data-view]")];
  for (const tab of tabs) {
    tab.onclick = () => {
      for (const item of tabs) item.setAttribute("aria-pressed", String(item === tab));
      for (const panel of section.querySelectorAll("[data-panel]")) panel.hidden = panel.dataset.panel !== tab.dataset.view;
    };
  }
}

let datasetLoaded, reportsLoaded;
function loadPage(page) {
  if (page === "check" || labPages.includes(page)) refreshModels();
  if (["lab", "dataset"].includes(page)) datasetLoaded ||= loadDataset().catch((error) => {
    showError("dataset-summary", error);
    datasetLoaded = null;
  });
  if (["lab", "report", "compare"].includes(page)) reportsLoaded ||= loadReports().catch((error) => {
    showError(page === "compare" ? "compare-scores" : page === "lab" ? "lab-summary" : "report-status", error);
    reportsLoaded = null;
  });
}
document.addEventListener("page-change", (event) => loadPage(event.detail));
loadPage(activePage);
