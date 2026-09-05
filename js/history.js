// 撤销/重做快照栈。快照为 cues 的深拷贝（structuredClone）。
export class History {
  constructor(limit = 100) {
    this.limit = limit;
    this.past = [];
    this.future = [];
  }

  push(snapshot) {
    this.past.push(snapshot);
    if (this.past.length > this.limit) this.past.shift();
    this.future = [];
  }

  undo(currentSnapshot) {
    if (!this.past.length) return null;
    this.future.push(currentSnapshot);
    return this.past.pop();
  }

  redo(currentSnapshot) {
    if (!this.future.length) return null;
    this.past.push(currentSnapshot);
    return this.future.pop();
  }

  clear() {
    this.past = [];
    this.future = [];
  }

  get canUndo() {
    return this.past.length > 0;
  }

  get canRedo() {
    return this.future.length > 0;
  }
}
