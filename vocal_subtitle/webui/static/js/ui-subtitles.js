(function() {
'use strict';
const part = {
    initSubtitleBatchControls() {
      var select = document.getElementById('batch-speaker-select');
      var newInput = document.getElementById('batch-speaker-new');
      var applyButton = document.getElementById('batch-speaker-apply');
      var mergeButton = document.getElementById('batch-merge-menu');
      var clearButton = document.getElementById('batch-clear-selection');
      if (!select || !newInput || !applyButton || !mergeButton || !clearButton) return;

      select.addEventListener('change', function() {
        newInput.style.display = this.value === '__new__' ? '' : 'none';
        if (this.value !== '__new__') newInput.value = '';
        App.ui.refreshBatchToolbar();
      });
      newInput.addEventListener('input', function() { App.ui.refreshBatchToolbar(); });
      applyButton.addEventListener('click', function() { App.ui.applyBatchSpeaker(); });
      mergeButton.addEventListener('click', function() {
        var menu = document.getElementById('batch-merge-options');
        if (menu) menu.style.display = menu.style.display === 'none' ? 'inline-flex' : 'none';
      });
      document.querySelectorAll('#batch-merge-options [data-separator]').forEach(function(button) {
        button.addEventListener('click', function() {
          App.ui.applyBatchMerge(this.dataset.separator);
        });
      });
      clearButton.addEventListener('click', function() { App.ui.clearSubtitleSelection(); });
    },

    clearBatchError() {
      var error = document.getElementById('subtitle-batch-error');
      if (error) { error.textContent = ''; error.style.display = 'none'; }
    },

    showBatchError(error) {
      var message = error && error.message ? error.message : String(error || '批量操作失败');
      try {
        var payload = JSON.parse(message);
        message = payload.detail || message;
      } catch (_) {}
      var errorEl = document.getElementById('subtitle-batch-error');
      if (errorEl) { errorEl.textContent = message; errorEl.style.display = ''; }
      toast(message, 'error');
    },

    clearSubtitleSelection(shouldRender) {
      App.state.selectedSubtitleIndexes.clear();
      App.state.selectionAnchorIndex = null;
      var menu = document.getElementById('batch-merge-options');
      if (menu) menu.style.display = 'none';
      App.ui.clearBatchError();
      if (shouldRender !== false) App.ui.renderTimeline(App.state.subtitleEvents);
      else App.ui.refreshBatchToolbar();
    },

    updateSelectionPresentation() {
      document.querySelectorAll('#subtitle-tbody tr[data-index]').forEach(function(row) {
        var index = parseInt(row.dataset.index, 10);
        var selected = App.state.selectedSubtitleIndexes.has(index);
        row.classList.toggle('subtitle-selected', selected);
        var checkbox = row.querySelector('.subtitle-select');
        if (checkbox) checkbox.checked = selected;
      });
      App.ui.refreshBatchToolbar();
    },

    cancelQueuedRowPlayback() {
      if (App.state.rowClickTimer !== null) {
        clearTimeout(App.state.rowClickTimer);
        App.state.rowClickTimer = null;
      }
    },

    queueRowPlayback(event) {
      App.ui.cancelQueuedRowPlayback();
      App.state.rowClickTimer = setTimeout(function() {
        App.state.rowClickTimer = null;
        var row = document.querySelector('tr[data-index="' + event.index + '"]');
        App.ui.playSegment(event, row ? row.querySelector('.play-btn') : null);
      }, 220);
    },

    stopCurrentPlayback() {
      if (App.state._audioPlayer) {
        App.state._audioPlayer.pause();
        App.state._audioPlayer = null;
      }
      document.querySelectorAll('.play-btn.playing').forEach(function(button) {
        button.classList.remove('playing');
      });
    },

    handleSubtitleKeyboard(event) {
      var target = event.target;
      var isField = target && target.closest && target.closest('input, textarea, select, [contenteditable="true"]');
      if (isField) return;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'a') {
        if (!App.state.subtitleEvents.length) return;
        event.preventDefault();
        App.state.selectedSubtitleIndexes = new Set(App.state.subtitleEvents.map(function(item) { return item.index; }));
        App.state.selectionAnchorIndex = App.state.subtitleEvents[0].index;
        App.ui.renderTimeline(App.state.subtitleEvents);
      } else if (event.key === 'Escape' && App.state.selectedSubtitleIndexes.size > 0) {
        event.preventDefault();
        App.ui.clearSubtitleSelection();
      }
    },

    handleSubtitleSelection(index, event) {
      var selected = App.state.selectedSubtitleIndexes;
      var events = App.state.subtitleEvents;
      var position = events.findIndex(function(item) { return item.index === index; });
      if (position < 0) return;

      if (event && event.shiftKey && App.state.selectionAnchorIndex !== null) {
        var anchor = events.findIndex(function(item) { return item.index === App.state.selectionAnchorIndex; });
        if (anchor >= 0) {
          selected.clear();
          var start = Math.min(anchor, position);
          var end = Math.max(anchor, position);
          for (var i = start; i <= end; i++) selected.add(events[i].index);
        }
      } else if (event && (event.ctrlKey || event.metaKey)) {
        if (selected.has(index)) selected.delete(index);
        else selected.add(index);
        App.state.selectionAnchorIndex = index;
      } else {
        selected.clear();
        selected.add(index);
        App.state.selectionAnchorIndex = index;
      }
      App.ui.updateSelectionPresentation();
    },

    selectedSubtitleIndexes() {
      return Array.from(App.state.selectedSubtitleIndexes).sort(function(a, b) { return a - b; });
    },

    speakerCandidates() {
      var candidates = new Map();
      App.state.subtitleEvents.forEach(function(event) {
        var label = (event.speaker_label || '').trim();
        var id = event.speaker_id === null || event.speaker_id === undefined ? null : event.speaker_id;
        if (!label && id === null) return;
        var display = label || ('说话人 ' + (id < 26 ? String.fromCharCode(65 + id) : id));
        var key = id === null ? 'label:' + encodeURIComponent(label) : 'id:' + id;
        if (!candidates.has(key)) candidates.set(key, {key: key, id: id, label: display});
      });
      return Array.from(candidates.values());
    },

    refreshBatchToolbar() {
      var toolbar = document.getElementById('subtitle-batch-toolbar');
      var count = document.getElementById('subtitle-batch-count');
      var select = document.getElementById('batch-speaker-select');
      var newInput = document.getElementById('batch-speaker-new');
      var applyButton = document.getElementById('batch-speaker-apply');
      var mergeButton = document.getElementById('batch-merge-menu');
      var mergeMenu = document.getElementById('batch-merge-options');
      if (!toolbar || !count || !select || !newInput || !applyButton || !mergeButton) return;

      var indexes = App.ui.selectedSubtitleIndexes();
      toolbar.style.display = indexes.length ? 'flex' : 'none';
      count.textContent = '已选择 ' + indexes.length + ' 条';
      mergeButton.disabled = indexes.length < 2;
      applyButton.disabled = !indexes.length || !select.value ||
        (select.value === '__new__' && !newInput.value.trim());
      if (!indexes.length && mergeMenu) mergeMenu.style.display = 'none';
    },

    renderBatchToolbar() {
      var select = document.getElementById('batch-speaker-select');
      if (!select) return;
      var previous = select.value;
      var html = '<option value="">选择说话人</option>';
      App.ui.speakerCandidates().forEach(function(candidate) {
        html += '<option value="' + App.ui._escapeAttr(candidate.key) + '">' + App.ui._escapeHtml(candidate.label) + '</option>';
      });
      html += '<option value="__new__">+ 添加说话人</option>';
      select.innerHTML = html;
      if (previous) {
        Array.from(select.options).some(function(option) {
          if (option.value === previous) { select.value = previous; return true; }
          return false;
        });
      }
      var newInput = document.getElementById('batch-speaker-new');
      if (newInput) newInput.style.display = select.value === '__new__' ? '' : 'none';
      App.ui.refreshBatchToolbar();
    },

    async applyBatchSpeaker() {
      var select = document.getElementById('batch-speaker-select');
      var newInput = document.getElementById('batch-speaker-new');
      if (!select || !select.value || !App.state.taskId) return;
      var payload = {action: 'speaker', indexes: App.ui.selectedSubtitleIndexes()};
      if (select.value === '__new__') {
        payload.speaker_label = (newInput.value || '').trim();
      } else if (select.value.indexOf('id:') === 0) {
        payload.speaker_id = parseInt(select.value.slice(3), 10);
        payload.speaker_label = select.options[select.selectedIndex].textContent;
      } else {
        payload.speaker_label = decodeURIComponent(select.value.slice(6));
      }
      try {
        var result = await App.api.batchEditSubtitles(App.state.taskId, payload);
        App.state.subtitleEvents = result.events || [];
        App.ui.clearSubtitleSelection(false);
        App.ui.renderTimeline(App.state.subtitleEvents);
        toast('已批量更新说话人并写入最终字幕', 'success');
      } catch (ex) {
        App.ui.showBatchError(ex);
      }
    },

    async applyBatchMerge(separator) {
      if (!App.state.taskId) return;
      try {
        var result = await App.api.batchEditSubtitles(App.state.taskId, {
          action: 'merge', indexes: App.ui.selectedSubtitleIndexes(), separator: separator
        });
        App.state.subtitleEvents = result.events || [];
        App.ui.clearSubtitleSelection(false);
        App.ui.renderTimeline(App.state.subtitleEvents);
        toast('字幕已合并并写入最终字幕', 'success');
      } catch (ex) {
        App.ui.showBatchError(ex);
      }
    },

    // Timeline
    renderTimeline(events) {
      const tbody = $('#subtitle-tbody');
      const thead = document.querySelector('.subtitle-table thead tr');
      const hasComparison = events.some(function(e) { return e.original_text; });
      const useCompare = hasComparison && App.state.subtitleViewMode !== 'simple';
      const finalFmt = App.state.subtitleFinalFormat;  // 'srt' | 'ass'

      // 更新表头（增加播放列）
      if (useCompare) {
        thead.innerHTML = '<th class="col-select">选</th><th class="col-idx">#</th><th class="col-time">开始</th><th class="col-time">结束</th><th class="col-speaker">说话人</th><th class="col-original">原文 (ASR)</th><th>优化后 (LLM)</th><th>最终版本 (' + finalFmt.toUpperCase() + ')</th><th class="col-play">播放</th>';
      } else {
        thead.innerHTML = '<th class="col-select">选</th><th class="col-idx">#</th><th class="col-time">开始</th><th class="col-time">结束</th><th class="col-speaker">说话人</th><th>最终版本 (' + finalFmt.toUpperCase() + ')</th><th class="col-play">播放</th>';
      }

      // 对比视图切换按钮 + 最终版本格式切换
      var toggleEl = document.getElementById('view-toggle');
      if (hasComparison && !toggleEl) {
        toggleEl = document.createElement('div');
        toggleEl.id = 'view-toggle';
        toggleEl.className = 'view-toggle';
        var tableWrap = document.querySelector('.subtitle-table-wrap');
        tableWrap.parentNode.insertBefore(toggleEl, tableWrap);
      } else if (!hasComparison && toggleEl) {
        toggleEl.remove();
        toggleEl = null;
      }

      // 重建 toggle bar 内容（含视图切换 + 格式切换）
      if (toggleEl) {
        toggleEl.innerHTML =
          '<span style="font-size:0.74rem;color:var(--text-secondary);">视图：</span>' +
          '<button class="view-toggle-btn ' + (App.state.subtitleViewMode !== 'simple' ? 'active' : '') + '" onclick="App.ui.switchView(\'compare\')">对比视图</button>' +
          '<button class="view-toggle-btn ' + (App.state.subtitleViewMode === 'simple' ? 'active' : '') + '" onclick="App.ui.switchView(\'simple\')">简洁视图</button>' +
          '<span class="final-format-toggle">' +
            '<span style="font-size:0.68rem;color:var(--text-tertiary);">最终格式：</span>' +
            '<button class="final-format-toggle-btn ' + (finalFmt === 'srt' ? 'active' : '') + '" onclick="App.ui.switchFinalFormat(\'srt\')">SRT</button>' +
            '<button class="final-format-toggle-btn ' + (finalFmt === 'ass' ? 'active' : '') + '" onclick="App.ui.switchFinalFormat(\'ass\')">ASS</button>' +
          '</span>';
      } else if (!hasComparison) {
        // 无对比数据时也有简洁视图 + 格式切换
        if (!toggleEl) {
          toggleEl = document.createElement('div');
          toggleEl.id = 'view-toggle';
          toggleEl.className = 'view-toggle';
          var tableWrap2 = document.querySelector('.subtitle-table-wrap');
          tableWrap2.parentNode.insertBefore(toggleEl, tableWrap2);
        }
        toggleEl.innerHTML =
          '<span style="font-size:0.74rem;color:var(--text-secondary);">视图：</span>' +
          '<button class="view-toggle-btn ' + (App.state.subtitleViewMode !== 'simple' ? 'active' : '') + '" onclick="App.ui.switchView(\'compare\')">对比视图</button>' +
          '<button class="view-toggle-btn active" onclick="App.ui.switchView(\'simple\')">简洁视图</button>' +
          '<span class="final-format-toggle">' +
            '<span style="font-size:0.68rem;color:var(--text-tertiary);">最终格式：</span>' +
            '<button class="final-format-toggle-btn ' + (finalFmt === 'srt' ? 'active' : '') + '" onclick="App.ui.switchFinalFormat(\'srt\')">SRT</button>' +
            '<button class="final-format-toggle-btn ' + (finalFmt === 'ass' ? 'active' : '') + '" onclick="App.ui.switchFinalFormat(\'ass\')">ASS</button>' +
          '</span>';
      }

      // 构建行
      tbody.innerHTML = '';
      events.forEach(function(e) {
        // 格式化说话人显示
        var speaker = '—';
        if (e.speaker_label) {
          speaker = App.ui.escapeHtml(e.speaker_label);
        } else if (e.speaker_id !== null && e.speaker_id !== undefined) {
          var letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
          var label = e.speaker_id < 26 ? letters[e.speaker_id] : String(e.speaker_id);
          speaker = '说话人 ' + label;
        }

        // 格式化最终版本
        var conflict = hasConflict(e, events);
        var formatted = finalFmt === 'ass' ? formatEventAsASS(e) : formatEventAsSRT(e);
        var finalHtml = App.ui.escapeHtml(formatted);
        // 存在真实冲突时，在文本前添加警告标记（但仍显示最终版本）
        if (conflict) {
          finalHtml = '<span class="conflict-badge" title="LLM 修改存在潜在冲突，请人工审核">⚠️ 待确认</span> ' + finalHtml;
        }

        const tr = document.createElement('tr');
        tr.dataset.index = String(e.index);
        if (App.state.selectedSubtitleIndexes.has(e.index)) tr.classList.add('subtitle-selected');
        var selectCell = '<td class="col-select"><input type="checkbox" class="subtitle-select" data-index="' + e.index + '" ' +
          (App.state.selectedSubtitleIndexes.has(e.index) ? 'checked ' : '') +
          'aria-label="选择字幕 ' + e.index + '"></td>';
        if (useCompare) {
          tr.innerHTML =
            selectCell + '<td class="col-idx">' + e.index + '</td>' +
            '<td class="col-time">' + formatTime(e.start) + '</td>' +
            '<td class="col-time">' + formatTime(e.end) + '</td>' +
            '<td class="col-speaker">' + speaker + '</td>' +
            '<td class="col-original" data-index="' + e.index + '">' + App.ui.escapeHtml(e.original_text || '') + '</td>' +
            '<td class="col-optimized" data-index="' + e.index + '">' + diffAndHighlight(e.original_text || '', e.text) + '</td>' +
            '<td class="col-final' + (conflict ? ' conflict' : '') + '" data-index="' + e.index + '">' + finalHtml + '</td>' +
            '<td class="col-play"><span class="play-btn" data-play="' + e.index + '" title="播放此段音频">▶</span></td>';
        } else {
          tr.innerHTML =
            selectCell + '<td class="col-idx">' + e.index + '</td>' +
            '<td class="col-time">' + formatTime(e.start) + '</td>' +
            '<td class="col-time">' + formatTime(e.end) + '</td>' +
            '<td class="col-speaker">' + speaker + '</td>' +
            '<td class="col-final' + (conflict ? ' conflict' : '') + '" data-index="' + e.index + '">' + finalHtml + '</td>' +
            '<td class="col-play"><span class="play-btn" data-play="' + e.index + '" title="播放此段音频">▶</span></td>';
        }

        // ---- Click / DblClick 交互模型 ----
        // 单击行任意位置 → 播放；双击文本列 → 编辑对应文本；
        // 双击时间/说话人区域 → 编辑该行最终字幕；播放按钮仅播放不编辑。
        //
        // 行级 click：先更新选区，再延迟播放；双击时由 dblclick 取消播放。
        // 行级 dblclick：停止播放，对非文本列区域打开最终字幕编辑。
        // 列级 dblclick：停止播放，对该列文本打开编辑。
        // 播放按钮：stopPropagation 阻止事件冒泡到行级。

        tr.addEventListener('click', function(ev) {
          var target = ev.target;
          if (target && target.closest && target.closest('.play-btn, input, button, select, textarea')) return;
          App.ui.handleSubtitleSelection(e.index, ev);
          App.ui.queueRowPlayback(e);
        });

        tr.addEventListener('dblclick', function(ev) {
          var target = ev.target;
          // 播放按钮 / 表单元素区域：消隐到行级不做任何事
          if (target && target.closest && target.closest('.play-btn, input, button, select, textarea')) return;
          // 文本列由各自的列级 dblclick 处理
          if (target && target.closest && target.closest('.col-original, .col-optimized, .col-final')) return;
          // 停止当前音频
          App.ui.cancelQueuedRowPlayback();
          App.ui.stopCurrentPlayback();
          var rowFinalCell = tr.querySelector('.col-final');
          if (rowFinalCell) App.ui.editSubtitle(rowFinalCell, e.index);
        });

        var finalCell = tr.querySelector('.col-final');
        if (finalCell) {
          finalCell.addEventListener('dblclick', function() {
            // 停止当前音频
            App.ui.cancelQueuedRowPlayback();
            App.ui.stopCurrentPlayback();
            App.ui.editSubtitle(finalCell, e.index);
          });
        }

        // 为原文和优化列也添加双击编辑（对比视图）
        if (useCompare) {
          var origCell = tr.querySelector('.col-original');
          var optCell = tr.querySelector('.col-optimized');
          if (origCell) origCell.addEventListener('dblclick', function() {
            App.ui.cancelQueuedRowPlayback();
            App.ui.stopCurrentPlayback();
            App.ui.editSubtitle(origCell, e.index);
          });
          if (optCell) optCell.addEventListener('dblclick', function() {
            App.ui.cancelQueuedRowPlayback();
            App.ui.stopCurrentPlayback();
            App.ui.editSubtitle(optCell, e.index);
          });
        }

        // 播放按钮：双击也仅播放，不进入编辑
        var playBtn = tr.querySelector('.play-btn');
        if (playBtn) {
          playBtn.addEventListener('click', function(ev) {
            ev.stopPropagation();
            App.ui.cancelQueuedRowPlayback();
            // 防抖：双击时避免重复触发播放
            var now = Date.now();
            var lastPlay = parseInt(this.getAttribute('data-last-play') || '0');
            if (now - lastPlay < 400) return;
            this.setAttribute('data-last-play', now);
            App.ui.playSegment(e, this);
          });
          playBtn.addEventListener('dblclick', function(ev) {
            ev.stopPropagation();
          });
        }

        var checkbox = tr.querySelector('.subtitle-select');
        if (checkbox) {
          checkbox.addEventListener('click', function(ev) { ev.stopPropagation(); });
          checkbox.addEventListener('change', function(ev) {
            App.ui.handleSubtitleSelection(e.index, ev);
          });
        }

        tbody.appendChild(tr);
      });
      App.ui.renderBatchToolbar();
    },

    switchView(mode) {
      App.state.subtitleViewMode = mode;
      App.ui.renderTimeline(App.state.subtitleEvents);
    },

    switchFinalFormat(fmt) {
      App.state.subtitleFinalFormat = fmt;
      App.ui.renderTimeline(App.state.subtitleEvents);
    },

    updateAudioExportBar(result) {
      var bar = document.getElementById('audio-export-bar');
      if (!bar) return;
      var hasVocals = !!(result && result.vocals_path);
      var hasAccomp = !!(result && result.accompaniment_path);
      if (hasVocals || hasAccomp) {
        bar.style.display = 'flex';
        // 如果没有人声或伴奏，隐藏对应按钮
        var btns = bar.querySelectorAll('.btn-export');
        if (btns[0]) btns[0].style.display = hasVocals ? '' : 'none';
        if (btns[1]) btns[1].style.display = hasAccomp ? '' : 'none';
      } else {
        bar.style.display = 'none';
      }
    },

    // 根据 LLM 优化是否生效，调整字幕下载栏的按钮布局
    // - LLM 启用时：隐藏主按钮，显示「干净版」+「LLM 优化版」
    // - LLM 未启用时：只显示主按钮「字幕文件」
    updateExportBar(result) {
      var mainBtn = document.getElementById('btn-srt-main');
      var cleanBtn = document.getElementById('btn-clean-srt');
      var llmBtn = document.getElementById('btn-llm-srt');
      var hasLLM = !!(result && result.llm_subtitle_path);

      if (mainBtn) mainBtn.style.display = hasLLM ? 'none' : '';
      if (cleanBtn) cleanBtn.style.display = hasLLM ? '' : 'none';
      if (llmBtn) llmBtn.style.display = hasLLM ? '' : 'none';
    },

    editSubtitle(cell, index) {
      if (cell.classList.contains('editing')) return;
      const event = App.state.subtitleEvents.find(function(e) { return e.index === index; });
      const originalText = event ? event.text : '';
      const isFinalCol = cell.classList.contains('col-final');
      cell.classList.add('editing');
      const input = document.createElement('input');
      input.type = 'text';
      input.name = 'subtitle-edit-' + index;
      input.autocomplete = 'off';
      input.setAttribute('autocomplete', 'off');
      input.setAttribute('data-lpignore', 'true');
      input.setAttribute('aria-autocomplete', 'none');
      input.setAttribute('role', 'textbox');
      input.setAttribute('data-form-type', 'other');
      input.placeholder = '编辑字幕文本…';
      input.setAttribute('aria-label', '编辑第' + index + '行字幕文本');
      input.value = originalText;
      cell.textContent = '';
      cell.appendChild(input);
      input.focus();
      input.select();

      const save = async () => {
        const newText = input.value.trim();
        cell.classList.remove('editing');
        if (newText && newText !== originalText) {
          try {
            await App.api.updateSubtitle(App.state.taskId, index, newText);
            if (event) { event.text = newText; event.original_text = null; }  // 手动编辑后清除冲突标记
            // 如果编辑的是最终版本列，重新格式化为 SRT/ASS
            if (isFinalCol && !hasConflict(event, App.state.subtitleEvents)) {
              var fmt = App.state.subtitleFinalFormat;
              cell.textContent = fmt === 'ass' ? formatEventAsASS(event) : formatEventAsSRT(event);
            } else {
              cell.textContent = newText;
            }
            toast('字幕 #' + index + ' 已更新并自动保存到文件', 'success');
          } catch (ex) {
            cell.textContent = isFinalCol ? (App.state.subtitleFinalFormat === 'ass' ? formatEventAsASS(event) : formatEventAsSRT(event)) : originalText;
            toast('更新失败: ' + ex.message, 'error');
          }
        } else {
          cell.textContent = isFinalCol ? (App.state.subtitleFinalFormat === 'ass' ? formatEventAsASS(event) : formatEventAsSRT(event)) : originalText;
        }
      };

      input.addEventListener('blur', save);
      input.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
        if (e.key === 'Escape') { input.value = originalText; input.blur(); }
      });
    },

    // Audio Playback
    playSegment(event, btnEl) {
      if (!App.state.taskId) {
        toast('请先完成处理或从历史记录加载任务', 'info');
        return;
      }

      // 停止当前播放
      if (App.state._audioPlayer) {
        App.state._audioPlayer.pause();
        App.state._audioPlayer = null;
      }
      // 重置所有播放按钮
      document.querySelectorAll('.play-btn.playing').forEach(function(c) { c.classList.remove('playing'); });

      var audio = document.getElementById('audio-player');
      var taskId = App.state.taskId;

      // 如果音频源未加载或任务不同，重新加载
      if (App.state._audioTaskId !== taskId) {
        var streamUrl = '/api/tasks/' + taskId + '/audio/stream?type=vocals';
        audio.src = streamUrl;
        audio.load();
        App.state._audioTaskId = taskId;
      }

      // 设置播放范围
      var startTime = event.start;
      var endTime = event.end;
      var duration = endTime - startTime;

      // 标记按钮为播放中
      if (btnEl) btnEl.classList.add('playing');

      // 高亮当前行
      var row = btnEl ? btnEl.closest('tr') : null;
      if (row) row.style.background = 'rgba(88,166,255,0.12)';

      var onTimeUpdate = function() {
        if (audio.currentTime >= endTime) {
          audio.pause();
          cleanup();
        }
      };

      var onError = function() {
        // Fallback: 尝试 input 类型
        if (audio.src.indexOf('type=input') === -1) {
          audio.src = '/api/tasks/' + taskId + '/audio/stream?type=input';
          audio.load();
          // 重试一次
          var retryOnce = function() {
            audio.removeEventListener('loadeddata', retryOnce);
            try {
              audio.currentTime = startTime;
              audio.play().catch(function() { cleanup(); });
            } catch(ex) { cleanup(); }
          };
          audio.addEventListener('loadeddata', retryOnce);
        } else {
          cleanup();
          toast('音频播放失败', 'error');
        }
      };

      var cleanup = function() {
        audio.removeEventListener('timeupdate', onTimeUpdate);
        audio.removeEventListener('error', onError);
        if (btnEl) btnEl.classList.remove('playing');
        if (row) row.style.background = '';
        App.state._audioPlayer = null;
      };

      audio.addEventListener('timeupdate', onTimeUpdate);
      audio.addEventListener('error', onError);

      App.state._audioPlayer = audio;

      // Seek and play
      var playWhenReady = function() {
        audio.removeEventListener('canplay', playWhenReady);
        audio.removeEventListener('loadeddata', playWhenReady);
        try {
          audio.currentTime = startTime;
          audio.play().catch(function() { cleanup(); });
        } catch(ex) { cleanup(); }
      };

      if (audio.readyState >= 2) {
        // Already has enough data
        try {
          audio.currentTime = startTime;
          audio.play().catch(function() { cleanup(); });
        } catch(ex) { cleanup(); }
      } else {
        audio.addEventListener('canplay', playWhenReady);
        audio.addEventListener('loadeddata', playWhenReady);
        audio.load();
      }
    },

    // History Rendering
};
window.VocalSubtitleUi = Object.assign(window.VocalSubtitleUi || {}, part);
})();

