const byId = (id) => document.getElementById(id);
const names = {identity: "Giữ nguyên đầu vào", baseline: "N-gram + Levenshtein", vit5: "ViT5-base", byt5: "ByT5-small"};
const sections = [...document.querySelectorAll("main > section")];
const node = (tag, value) => Object.assign(document.createElement(tag), {textContent: value});

function route() {
  const key = location.hash.slice(1);
  const active = sections.some((section) => section.id === key) ? key : "home";
  sections.forEach((section) => { section.hidden = section.id !== active; });
  document.querySelectorAll("nav a").forEach((link) => {
    if (link.hash === "#" + active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
}
window.addEventListener("hashchange", route);
route();

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
input.addEventListener("invalid", () => {
  if (input.validity.valueMissing) input.setCustomValidity("Nhập văn bản tiếng Việt để kiểm tra.");
});
function count() {
  byId("syllables").textContent = (input.value.trim() ? (input.value.match(/[\p{L}\p{M}\p{N}]+/gu) || []).length : 0) + " âm tiết";
}
let editRequest = 0;
let checked = null;
byId("check-model").addEventListener("change", invalidateCheck);

function invalidateCheck() {
  editRequest++;
  checked = null;
  byId("check-result").hidden = true;
  byId("check-status").textContent = "";
  byId("suggestions").replaceChildren();
}

async function copyText(text, status) {
  try {
    await navigator.clipboard.writeText(text);
    status.textContent = "Đã sao chép văn bản.";
  } catch {
    status.textContent = "Trình duyệt chưa cho phép sao chép.";
  }
}
input.addEventListener("input", () => {
  invalidateCheck();
  input.setCustomValidity("");
  count();
});
count();
byId("reset").addEventListener("click", () => {
  input.value = "";
  input.dispatchEvent(new Event("input"));
  input.focus();
});

onForm("check-form", "check-status", async (submitter) => {
  const automatic = submitter?.value === "apply";
  const text = input.value;
  const model = byId("check-model").value;
  if (!model) throw new Error("Chưa có mô hình sẵn sàng.");
  if (automatic && checked?.text === text && checked.model === model) {
    checked.applyAutomatic();
    return;
  }

  invalidateCheck();
  byId("check-status").textContent = "Đang xử lý…";
  const current = editRequest;
  let result;
  try {
    result = await request("/api/check", {text, model});
  } catch (error) {
    if (editRequest === current) throw error;
    return;
  }
  if (editRequest !== current) return;

  byId("syllables").textContent = result.syllable_count + " âm tiết";
  const edits = result.errors.filter((edit) => edit.suggestion !== edit.original);
  const choices = new Map(edits.map((edit) => [edit.id, edit.suggestion]));
  const state = new Map(edits.map((edit) => [edit.id, "pending"]));
  const removed = new Set();
  const applyPending = (eligible) => {
    edits.forEach((edit) => {
      if (state.get(edit.id) === "pending" && eligible(edit)) state.set(edit.id, "applied");
    });
    update();
  };

  function update() {
    const accepted = new Map(edits.filter((edit) => state.get(edit.id) === "applied")
      .map((edit) => [edit.id, choices.get(edit.id)]));
    annotate(result.original_text, edits.filter((edit) => !removed.has(edit.id)), new Map(), byId("annotation"), true);
    annotate(result.original_text, edits, accepted, byId("corrected"), false);
    const counts = {pending: 0, applied: 0, ignored: 0};
    state.forEach((value) => { counts[value]++; });
    byId("apply-all").disabled = counts.pending === 0;
    byId("undo-all").disabled = counts.applied === 0;
    byId("copy-status").textContent = "";
    const status = edits.length
      ? ["Phát hiện " + edits.length, "Đã sửa " + counts.applied, "Chờ xử lý " + counts.pending, "Bỏ qua " + counts.ignored].join(" · ")
      : "Không phát hiện lỗi chính tả";
    byId("check-status").textContent = status + " · " + names[result.model] + " · Xử lý " + result.latency_ms + " ms";
    byId("suggestions").replaceChildren(...edits.filter((edit) => !removed.has(edit.id)).map((edit) => {
      const id = edit.id;
      const li = node("li", "");
      li.id = "suggestion-" + id;
      const current = state.get(id);
      li.append(node("span", edit.original + " → "));
      const candidates = [edit.suggestion, ...edit.alternatives.filter((option) => option !== edit.original)];
      if (candidates.length > 1) {
        const select = node("select", "");
        select.setAttribute("aria-label", "Phương án sửa vị trí " + edit.position);
        select.append(...candidates.map((value) => {
          const option = node("option", value);
          option.value = value;
          return option;
        }));
        select.value = choices.get(id);
        select.disabled = current === "ignored";
        select.onchange = () => {
          choices.set(id, select.value);
          update();
          byId(li.id).querySelector("select").focus();
        };
        li.append(select);
      } else {
        li.append(node("strong", edit.suggestion));
      }
      li.append(node("span", " · " + edit.category + " · Vị trí " + edit.position));
      if (!edit.auto_apply) li.append(node("span", " · Cần xem lại"));
      const toggle = node("button", current === "pending" ? "Áp dụng" : "Hoàn tác");
      toggle.type = "button";
      toggle.className = "secondary";
      toggle.onclick = () => {
        state.set(id, current === "pending" ? "applied" : "pending");
        update();
        byId(li.id).querySelector("button").focus();
      };
      li.append(toggle);
      if (current === "pending") {
        for (const label of ["Bỏ qua", "Xóa gợi ý"]) {
          const button = node("button", label);
          button.type = "button";
          button.className = "secondary";
          button.onclick = () => {
            state.set(id, "ignored");
            if (label === "Xóa gợi ý") removed.add(id);
            update();
          };
          li.append(button);
        }
      }
      return li;
    }));
  }

  const applyAutomatic = () => applyPending((edit) => edit.auto_apply && choices.get(edit.id) === edit.suggestion);
  checked = {text, model, applyAutomatic};
  byId("apply-all").onclick = () => applyPending(() => true);
  byId("undo-all").onclick = () => {
    state.forEach((status, id) => { if (status === "applied") state.set(id, "pending"); });
    update();
  };
  byId("copy-result").onclick = () => copyText(byId("corrected").textContent, byId("copy-status"));
  if (automatic) applyAutomatic(); else update();
  byId("check-result").hidden = false;
});
