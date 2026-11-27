const byId = (id) => document.getElementById(id);
const names = {identity: "Giữ nguyên đầu vào", baseline: "N-gram + Levenshtein", vit5: "ViT5-base", byt5: "ByT5-small"};
const sections = [...document.querySelectorAll("main > section")];
const node = (tag, value) => Object.assign(document.createElement(tag), {textContent: value});
let maximumText = 4000;

function checkedInput(text) {
  if (!text.trim()) throw new Error("Nhập văn bản tiếng Việt để kiểm tra.");
  if (Array.from(text).length > maximumText) {
    throw new Error("Văn bản vượt " + maximumText + " ký tự. Hãy chia thành các đoạn nhỏ để xử lý.");
  }
  return text;
}

const pagePaths = {home: "/", check: "/editor", guide: "/guide", about: "/about", lab: "/lab",
  compare: "/lab/compare", dataset: "/lab/dataset", noise: "/lab/augmentation", report: "/lab/experiments", models: "/lab/models"};
const labPages = ["lab", "compare", "dataset", "noise", "report", "models"];
let activePage;

function route() {
  const legacy = location.hash.slice(1);
  const path = new URL(location.href).pathname;
  const legacyPage = path === "/" && Object.hasOwn(pagePaths, legacy);
  const active = legacyPage ? legacy : Object.keys(pagePaths).find((key) => pagePaths[key] === path) || "home";
  if (legacyPage) history.replaceState(null, "", pagePaths[active] + location.search);
  activePage = active;
  const lab = labPages.includes(active);
  byId("main").classList.toggle("lab-layout", lab);
  byId("lab-nav").hidden = !lab;
  byId("breadcrumb").hidden = !lab || active === "lab";
  sections.forEach((section) => { section.hidden = section.id !== active; });
  const heading = byId(active).querySelector("h1").textContent.trim();
  byId("breadcrumb-page").textContent = heading;
  document.title = active === "home" ? "VietSpell AI · Kiểm tra chính tả tiếng Việt" : heading + " · VietSpell AI";
  document.querySelectorAll("nav a").forEach((link) => {
    if (link.pathname === pagePaths[active]) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  document.dispatchEvent(new CustomEvent("page-change", {detail: active}));
}

function navigate(key) {
  if (location.pathname !== pagePaths[key]) {
    history.pushState(null, "", pagePaths[key]);
    route();
    window.scrollTo(0, 0);
  }
}

document.addEventListener("click", (event) => {
  const link = event.target.closest("a");
  if (!link || event.defaultPrevented || event.button || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey ||
      link.origin !== location.origin || link.hash || link.search || link.target || link.hasAttribute("download")) return;
  const key = Object.keys(pagePaths).find((id) => pagePaths[id] === link.pathname);
  if (!key) return;
  event.preventDefault();
  navigate(key);
  byId("main").focus({preventScroll: true});
});
window.addEventListener("popstate", route);
window.addEventListener("hashchange", route);
route();

function preference(key, value) {
  try {
    if (value === undefined) return localStorage.getItem("vietspell-" + key);
    if (value === null) localStorage.removeItem("vietspell-" + key);
    else localStorage.setItem("vietspell-" + key, value);
  } catch { return null; }
}

function setTheme(dark) {
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  byId("theme").textContent = dark ? "Giao diện sáng" : "Giao diện tối";
  const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  for (const [key, value] of Object.entries({width: "16", height: "16", viewBox: "0 0 24 24",
    fill: "none", stroke: "currentColor", "stroke-width": "2", "stroke-linecap": "round",
    "stroke-linejoin": "round", "aria-hidden": "true"})) icon.setAttribute(key, value);
  const shape = document.createElementNS(icon.namespaceURI, "path");
  shape.setAttribute("d", dark
    ? "M12 2v2m0 16v2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42 M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z"
    : "M20.985 12.486a9 9 0 1 1-9.473-9.472c.405-.022.617.46.402.803a6.5 6.5 0 0 0 8.268 8.268c.344-.215.825-.003.803.401");
  icon.append(shape);
  byId("theme").prepend(icon);
  byId("theme").setAttribute("aria-pressed", String(dark));
}
setTheme(preference("theme") === "dark");
const menuToggle = byId("menu-toggle");
const primaryNav = byId("primary-nav");
function setMenu(open) {
  menuToggle.setAttribute("aria-expanded", String(open));
  primaryNav.classList.toggle("menu-open", open);
}
menuToggle.onclick = () => setMenu(menuToggle.getAttribute("aria-expanded") !== "true");
document.addEventListener("click", (event) => {
  if (event.target.closest("#primary-nav a") || !event.target.closest("#primary-nav, #menu-toggle")) setMenu(false);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && menuToggle.getAttribute("aria-expanded") === "true") {
    setMenu(false);
    menuToggle.focus();
  }
});
byId("theme").onclick = () => {
  const dark = document.documentElement.dataset.theme !== "dark";
  setTheme(dark);
  preference("theme", dark ? "dark" : "light");
};

async function request(path, body) {
  const options = body ? {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)} : undefined;
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Yêu cầu thất bại.");
  return data;
}

function onForm(id, statusId, action) {
  const form = byId(id);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const buttons = [...form.querySelectorAll('[type="submit"]')];
    buttons.forEach((button) => { button.disabled = true; });
    byId(statusId).textContent = "Đang xử lý…";
    try { await action(event.submitter); }
    catch (error) { byId(statusId).textContent = error.message; }
    finally { buttons.forEach((button) => { button.disabled = false; }); }
  });
}

function annotate(text, edits, selected, target, showOriginal) {
  const chars = Array.from(text);
  const part = (start, end) => chars.slice(start, end).join("");
  const contents = [];
  let position = 0;
  edits.forEach((edit) => {
    contents.push(document.createTextNode(part(position, edit.start)));
    const original = part(edit.start, edit.end);
    const value = selected.get(edit.id);
    if (showOriginal || value !== undefined) {
      const mark = node(showOriginal && target.id === "annotation" ? "button" : "mark", showOriginal ? original : value);
      if (!showOriginal) mark.className = "applied";
      if (mark.tagName === "BUTTON") {
        mark.type = "button";
        mark.className = "original-mark";
        mark.setAttribute("aria-label", original + " · Xem gợi ý");
        mark.onclick = () => byId("suggestion-" + edit.id)?.scrollIntoView({block: "center"});
      }
      contents.push(mark);
    } else {
      contents.push(document.createTextNode(original));
    }
    position = edit.end;
  });
  contents.push(document.createTextNode(part(position, chars.length)));
  target.replaceChildren(...contents);
}

const input = byId("check-text");
const selector = byId("check-model");
const popover = byId("edit-popover");
const editorTools = document.querySelector(".editor-tools");
const mobile = matchMedia("(max-width: 680px)");
const editorSample = "Hôm nai tôi đến chường để học lập chình. Tôi rất thít ngành công nghệ thông tin.";
const suggestionFilters = [...document.querySelectorAll("[data-suggestion-status]")];
const bulkApply = [byId("apply-all"), byId("apply-all-sidebar")];
const bulkUndo = [byId("undo-all"), byId("undo-all-sidebar")];
let editRequest = 0;
let checked = null;
let pending = false;
let timer;
let toastTimer;

document.addEventListener("click", (event) => {
  if (editorTools.open && !editorTools.contains(event.target)) editorTools.open = false;
});
editorTools.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && editorTools.open) {
    editorTools.open = false;
    editorTools.querySelector("summary").focus();
  }
});

function closeEditPopover() {
  if (popover.matches(":popover-open")) popover.hidePopover();
}
byId("check-result").addEventListener("scroll", closeEditPopover, {passive: true});
window.addEventListener("scroll", closeEditPopover, {passive: true});

function toast(message) {
  byId("copy-status").textContent = message;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { byId("copy-status").textContent = ""; }, 2800);
}

async function copyText(text, status) {
  try {
    await navigator.clipboard.writeText(text);
    status.textContent = "Đã sao chép văn bản.";
  } catch {
    status.textContent = "Trình duyệt chưa cho phép sao chép.";
  }
}

function setSheet(open) {
  byId("suggestion-panel").classList.toggle("sheet-open", open);
  byId("open-suggestions").setAttribute("aria-expanded", String(open));
}

function editorStatus(message) {
  byId("check-status").textContent = message;
  byId("mobile-check-status").textContent = message;
}

function setReview(show) {
  closeEditPopover();
  input.hidden = show;
  byId("check-result").hidden = !show;
  byId("review-view").setAttribute("aria-pressed", String(show));
  byId("edit-view").setAttribute("aria-pressed", String(!show));
}

function displayedText() {
  return checked?.output ?? input.value;
}

function count() {
  const text = displayedText();
  byId("syllables").textContent = (text.match(/[\p{L}\p{M}\p{N}]+/gu) || []).length + " âm tiết · " +
    Array.from(text).length + " ký tự";
}

function invalidateCheck(message) {
  editRequest++;
  checked = null;
  setReview(false);
  byId("review-view").disabled = true;
  byId("editor-counts").hidden = true;
  byId("suggestion-pages").hidden = byId("suggestion-actions").hidden = true;
  byId("open-suggestions").textContent = "Xem gợi ý";
  setSheet(false);
  byId("suggestions").replaceChildren();
  [...bulkApply, ...bulkUndo, byId("undo-change"), byId("redo-change")].forEach((button) => { button.disabled = true; });
  editorStatus(message || (input.value.trim() ? "Bấm Kiểm tra để xem gợi ý." : "Nhập hoặc dán văn bản để bắt đầu."));
  count();
}

function tryModel(key) {
  const valid = Array.from(selector.options).some((option) => option.value === key && !option.disabled);
  selector.value = valid ? key : "auto";
  selector.dispatchEvent(new Event("change"));
  navigate("check");
}

function scheduleCheck() {
  clearTimeout(timer);
  if (byId("auto-check").checked && input.value.trim() && activePage === "check") {
    timer = setTimeout(() => {
      if (pending) scheduleCheck();
      else checkDocument();
    }, 850);
  }
}

function updateDraft() {
  if (byId("save-draft").checked) preference("draft", displayedText());
}

function replaceText(text) {
  input.value = text;
  input.dispatchEvent(new Event("input"));
  input.focus();
}

selector.addEventListener("change", () => {
  input.value = displayedText();
  invalidateCheck();
  updateDraft();
});
byId("save-draft").checked = preference("save-draft") === "yes";
if (byId("save-draft").checked) input.value = preference("draft") || "";
if (new URLSearchParams(location.search).has("sample") && activePage === "check") input.value = editorSample;
invalidateCheck();
input.addEventListener("input", () => {
  const stale = Boolean(checked || pending);
  invalidateCheck(stale && input.value.trim() ? "Văn bản đã thay đổi. Cần kiểm tra lại." : undefined);
  updateDraft();
  scheduleCheck();
});
byId("sample-text").onclick = () => replaceText(editorSample);
byId("reset").onclick = () => input.value ? byId("clear-dialog").showModal() : replaceText("");
byId("clear-dialog").addEventListener("close", () => {
  if (byId("clear-dialog").returnValue === "clear") replaceText("");
});
byId("edit-view").onclick = () => {
  if (checked && checked.output !== input.value) replaceText(checked.output);
  else { setReview(false); input.focus(); }
};
byId("review-view").onclick = () => setReview(true);
byId("open-suggestions").onclick = () => { setSheet(true); byId("close-suggestions").focus(); };
byId("close-suggestions").onclick = () => { setSheet(false); byId("open-suggestions").focus(); };
byId("suggestion-panel").addEventListener("keydown", (event) => {
  if (event.key === "Escape" && mobile.matches) byId("close-suggestions").click();
});
byId("editor-settings").onclick = () => byId("settings-dialog").showModal();
byId("close-settings").onclick = () => byId("settings-dialog").close();
byId("auto-check").onchange = scheduleCheck;
byId("save-draft").onchange = () => {
  preference("save-draft", byId("save-draft").checked ? "yes" : null);
  if (!byId("save-draft").checked) preference("draft", null);
  else updateDraft();
};
byId("open-text").onclick = () => byId("text-file").click();
byId("text-file").onchange = async () => {
  const file = byId("text-file").files[0];
  byId("text-file").value = "";
  if (!file) return;
  try {
    if (file.size > maximumText * 4 + 3) throw new Error("Tệp TXT vượt giới hạn văn bản.");
    replaceText(checkedInput(await file.text()));
  } catch (error) { toast(error.message); }
};
byId("paste-text").onclick = async () => {
  try { replaceText(await navigator.clipboard.readText()); }
  catch { toast("Dùng Ctrl+V hoặc lệnh Dán của thiết bị."); }
};
byId("fullscreen").onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await byId("check").requestFullscreen();
  } catch { toast("Trình duyệt chưa hỗ trợ toàn màn hình."); }
};
document.addEventListener("fullscreenchange", () => {
  byId("fullscreen").textContent = document.fullscreenElement ? "Thoát toàn màn hình" : "Toàn màn hình";
});
byId("copy-result").onclick = async () => {
  if (!displayedText()) return;
  try { await navigator.clipboard.writeText(displayedText()); toast("Đã sao chép văn bản."); }
  catch { toast("Trình duyệt chưa cho phép sao chép."); }
};
byId("export-text").onclick = () => {
  if (!displayedText()) return;
  download("vietspell.txt", displayedText(), "text/plain;charset=utf-8");
  toast("Đã xuất văn bản TXT.");
};

byId("check-form").addEventListener("submit", (event) => {
  event.preventDefault();
  clearTimeout(timer);
  checkDocument();
});

byId("auto-apply").onclick = () => {
  if (checked) checked.applyAutomatic();
  else checkDocument(true);
};

async function checkDocument(automatic = false) {
  if (pending) return;
  const text = displayedText();
  const model = selector.value || "auto";
  try { checkedInput(text); }
  catch (error) { editorStatus(error.message); return; }
  input.value = text;
  invalidateCheck();
  const revision = editRequest;
  const submit = byId("check-form").querySelector("[type=submit]");
  pending = submit.disabled = true;
  editorStatus("Đang phân tích văn bản và ngữ cảnh…");
  try {
    const result = await request("/api/check", {text, model});
    if (revision !== editRequest) return;
    showEdits(result);
    byId("check-model-info").textContent = "Vừa kiểm tra bằng " + names[result.model] + ".";
    setReview(true);
    if (automatic) checked.applyAutomatic();
  } catch (error) {
    if (revision === editRequest) editorStatus(error.message === "Failed to fetch"
      ? "Không kết nối được máy chủ. Vui lòng thử lại." : error.message);
  } finally {
    pending = submit.disabled = false;
  }
}

function showEdits(result) {
  const text = result.original_text;
  const chars = Array.from(text);
  const edits = result.errors.filter((edit) => edit.suggestion !== edit.original);
  const history = [edits.map((edit) => ({status: "pending", choice: edit.suggestion}))];
  let cursor = 0;
  let filter = "pending";
  let expanded = null;
  let page = 0;
  const pageSize = 24;
  const list = byId("suggestions");
  const scroll = byId("suggestion-scroll");
  checked = {output: text};

  function commit(change) {
    const next = history[cursor].map((item) => ({...item}));
    change(next);
    history.splice(cursor + 1, Infinity, next);
    cursor++;
    render();
  }

  function focusEdit(index) {
    setReview(true);
    if (mobile.matches) setSheet(false);
    const mark = byId("error-mark-" + index);
    mark?.scrollIntoView({block: "nearest"});
    mark?.focus({preventScroll: true});
  }

  function controls(edit, index, inline = false) {
    const entry = history[cursor][index];
    const actions = node("div", "");
    actions.className = "controls";
    const options = [edit.suggestion, ...edit.alternatives.filter((value) => value !== edit.original)];
    if (options.length > 1 && entry.status !== "ignored") {
      const variants = node("div", "");
      variants.className = "variant-options";
      variants.setAttribute("role", "group");
      variants.setAttribute("aria-label", "Phương án thay thế cho " + edit.original);
      options.forEach((value, optionIndex) => {
        const option = Object.assign(node("button", value), {type: "button", className: "secondary"});
        option.setAttribute("aria-pressed", String(value === entry.choice));
        option.title = optionIndex === 0 ? "Phương án được đề xuất" : "Phương án cùng dạng chữ";
        option.onclick = () => {
          if (value === entry.choice) return;
          commit((next) => { next[index].choice = value; });
          if (inline) byId("error-mark-" + index)?.click();
          const container = inline ? popover : byId("suggestion-" + index);
          container?.querySelector(".variant-options button[aria-pressed=true]")?.focus();
        };
        variants.append(option);
      });
      actions.append(variants);
    }
    function act(label, status) {
      const control = Object.assign(node("button", label), {type: "button", className: "secondary"});
      control.onclick = () => {
        commit((next) => { next[index].status = status; });
        if (!inline) (list.querySelector(".suggestion-row") || suggestionFilters.find((item) =>
          item.dataset.suggestionStatus === filter))?.focus({preventScroll: true});
      };
      return control;
    }
    actions.append(...(entry.status === "pending"
      ? [act("Áp dụng", "applied"), act("Bỏ qua", "ignored")]
      : [act(entry.status === "applied" ? "Hoàn tác" : "Khôi phục gợi ý", "pending")]));
    return actions;
  }

  function openPopover(mark, edit, index) {
    popover.replaceChildren(node("strong", edit.original + " → " + history[cursor][index].choice),
      node("p", edit.category), controls(edit, index, true));
    popover.showPopover();
    const box = mark.getBoundingClientRect();
    const width = popover.offsetWidth;
    popover.style.left = Math.max(8, Math.min(box.left, innerWidth - width - 8)) + "px";
    popover.style.top = (box.bottom + popover.offsetHeight + 8 < innerHeight ? box.bottom + 5
      : Math.max(8, box.top - popover.offsetHeight - 5)) + "px";
    popover.querySelector("button")?.focus();
  }

  function render() {
    closeEditPopover();
    const entries = history[cursor];
    const counts = {pending: 0, applied: 0, ignored: 0};
    const contents = [];
    let position = 0;
    edits.forEach((edit, index) => {
      const {status, choice} = entries[index];
      counts[status]++;
      contents.push(document.createTextNode(chars.slice(position, edit.start).join("")));
      const mark = node(status === "pending" ? "button" : status === "applied" ? "mark" : "span",
        status === "applied" ? choice : edit.original);
      mark.id = "error-mark-" + index;
      if (status === "pending") {
        mark.type = "button";
        mark.className = "original-mark";
        mark.setAttribute("aria-label", "Lỗi chính tả: " + edit.original + ". Xem gợi ý");
        mark.onclick = () => openPopover(mark, edit, index);
      } else {
        mark.tabIndex = -1;
        if (status === "applied") mark.className = "applied";
      }
      contents.push(mark);
      position = edit.end;
    });
    contents.push(document.createTextNode(chars.slice(position).join("")));
    byId("annotation").replaceChildren(...contents);
    checked.output = byId("annotation").textContent;
    for (const button of suggestionFilters) button.querySelector("span").textContent = counts[button.dataset.suggestionStatus];
    byId("editor-counts").hidden = !edits.length;
    byId("suggestion-actions").hidden = !edits.length;
    byId("open-suggestions").textContent = "Xem gợi ý (" + counts.pending + ")";
    editorStatus((edits.length === 0 ? "Không phát hiện lỗi chính tả."
      : counts.pending ? "Chọn từ được đánh dấu hoặc một gợi ý để xem cách sửa."
      : counts.ignored ? "Đã xử lý các gợi ý." : "Đã áp dụng tất cả gợi ý chỉnh sửa.")
      + (result.warning ? " " + result.warning : ""));
    renderSuggestions();
    for (const button of bulkApply) button.disabled = !counts.pending;
    for (const button of bulkUndo) button.disabled = !counts.applied;
    byId("undo-change").disabled = cursor === 0;
    byId("redo-change").disabled = cursor === history.length - 1;
    byId("review-view").disabled = false;
    count();
    updateDraft();
  }

  function renderSuggestions() {
    const entries = history[cursor];
    const matches = entries.flatMap((entry, index) => entry.status === filter ? [index] : []);
    const pages = Math.max(1, Math.ceil(matches.length / pageSize));
    page = Math.min(page, pages - 1);
    const shown = matches.slice(page * pageSize, (page + 1) * pageSize);
    if (!shown.includes(expanded)) expanded = null;
    list.replaceChildren(...shown.map((index) => {
      const edit = edits[index];
      const item = node("li", "");
      item.id = "suggestion-" + index;
      const toggle = Object.assign(node("button", ""), {type: "button", className: "suggestion-row"});
      toggle.append(node("span", edit.original + " → " + entries[index].choice), node("small", "Vị trí " + edit.position));
      toggle.setAttribute("aria-expanded", String(expanded === index));
      toggle.onclick = () => {
        expanded = expanded === index ? null : index;
        renderSuggestions();
        byId(item.id).querySelector(".suggestion-row").focus({preventScroll: true});
      };
      item.append(toggle);
      if (expanded !== index) return item;
      const detail = node("div", "");
      detail.className = "suggestion-detail";
      const context = node("p", "");
      context.className = "suggestion-context";
      const from = Math.max(0, edit.start - 26);
      const to = Math.min(chars.length, edit.end + 26);
      context.append(document.createTextNode((from ? "…" : "") + chars.slice(from, edit.start).join("")), node("strong", entries[index].choice),
        document.createTextNode(chars.slice(edit.end, to).join("") + (to < chars.length ? "…" : "")));
      const category = edit.source === "accent_support" ? "Gợi ý thêm về dấu" : edit.category;
      const jump = Object.assign(node("button", "Đến vị trí"), {type: "button", className: "suggestion-jump"});
      jump.onclick = () => focusEdit(index);
      detail.append(context, node("p", category + (filter === "pending" && !edit.auto_apply ? " · Cần xác nhận" : "")),
        controls(edit, index), jump);
      item.append(detail);
      return item;
    }));
    if (edits.length && !matches.length && filter !== "pending") {
      list.append(Object.assign(node("li", filter === "applied" ? "Chưa có gợi ý đã sửa." : "Chưa có gợi ý đã bỏ qua."),
        {className: "suggestion-empty"}));
    }
    for (const button of suggestionFilters) button.setAttribute("aria-pressed", String(button.dataset.suggestionStatus === filter));
    byId("suggestion-pages").hidden = pages < 2;
    byId("suggestions-prev").disabled = page === 0;
    byId("suggestions-next").disabled = page + 1 >= pages;
    byId("suggestions-page").textContent = (page + 1) + " / " + pages;
  }

  for (const button of suggestionFilters) button.onclick = () => {
    filter = button.dataset.suggestionStatus;
    expanded = null;
    page = scroll.scrollTop = 0;
    renderSuggestions();
  };
  for (const [id, delta] of [["suggestions-prev", -1], ["suggestions-next", 1]]) byId(id).onclick = () => {
    page += delta;
    expanded = null;
    scroll.scrollTop = 0;
    renderSuggestions();
  };
  const applyAll = () => {
    commit((items) => items.forEach((item) => { if (item.status === "pending") item.status = "applied"; }));
    setReview(true);
    toast("Đã sửa các gợi ý còn lại.");
  };
  const undoAll = () => {
    commit((items) => items.forEach((item) => { if (item.status === "applied") item.status = "pending"; }));
    toast("Đã hoàn tác các chỉnh sửa.");
  };
  for (const button of bulkApply) button.onclick = applyAll;
  for (const button of bulkUndo) button.onclick = undoAll;
  checked.applyAutomatic = () => {
    const eligible = (item, index) => item.status === "pending" && edits[index].auto_apply
      && item.choice === edits[index].suggestion;
    if (!history[cursor].some(eligible)) {
      toast("Không có gợi ý đủ điều kiện sửa tự động.");
      return;
    }
    commit((items) => items.forEach((item, index) => { if (eligible(item, index)) item.status = "applied"; }));
    setReview(true);
  };
  byId("undo-change").onclick = () => { if (cursor) { cursor--; render(); toast("Đã hoàn tác."); } };
  byId("redo-change").onclick = () => { if (cursor < history.length - 1) { cursor++; render(); toast("Đã làm lại."); } };
  render();
}
