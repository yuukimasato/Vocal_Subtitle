// 撤销/重做快照栈。快照为 cues 的深拷贝（structuredClone）。
// push 支持可选 coalesceKey：与上一次入栈同键且中间没有其他入栈时，
// 替换栈顶而不是新入栈（对齐 Aegisub 自动提交的合并语义——一整段拖拽只算一步撤销）。
export class History {
  constructor(limit = 100) {
    this.limit = limit;
    this.past = [];
    this.future = [];
    this.lastKey = null;
    // 干净基线 = 导出/打开文件时的 past 深度；撤销回到基线即内容与文件一致
    this.baseline = 0;
  }

  push(snapshot, { coalesceKey = null } = {}) {
    // 同键连续提交：栈顶已是这一链条起点的「修改前」快照，直接沿用（一步撤销回到起点）
    if (this.wouldCoalesce(coalesceKey)) {
      this.future = [];
      return;
    }
    this.past.push(snapshot);
    if (this.past.length > this.limit) {
      this.past.shift();
      if (this.baseline > 0) this.baseline -= 1;
    }
    this.future = [];
    this.lastKey = coalesceKey;
  }

  // 与 push 的合并判定完全一致，供调用方在深拷贝前先行判断（省掉注定被丢弃的快照）
  wouldCoalesce(coalesceKey) {
    return coalesceKey !== null && coalesceKey === this.lastKey && this.past.length > 0;
  }

  undo(currentSnapshot) {
    if (!this.past.length) return null;
    this.future.push(currentSnapshot);
    this.lastKey = null;
    return this.past.pop();
  }

  redo(currentSnapshot) {
    if (!this.future.length) return null;
    this.past.push(currentSnapshot);
    this.lastKey = null;
    return this.future.pop();
  }

  clear() {
    this.past = [];
    this.future = [];
    this.lastKey = null;
    this.baseline = 0;
  }

  // 把当前状态标记为「与文件一致」（打开/导出成功时调用）
  markBaseline() {
    this.baseline = this.past.length;
  }

  get isDirty() {
    return this.past.length !== this.baseline;
  }

  get canUndo() {
    return this.past.length > 0;
  }

  get canRedo() {
    return this.future.length > 0;
  }
}
