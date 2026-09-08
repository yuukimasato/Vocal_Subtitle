import test from 'node:test';
import assert from 'node:assert/strict';
import {
  parseTimestamp,
  formatSrtTime,
  formatVttTime,
  formatAssTime,
  formatClock,
  parseFlexibleTime,
} from '../js/format/time.js';

test('parseTimestamp 接受常见时间戳形式', () => {
  assert.equal(parseTimestamp('00:00:01,500'), 1.5);
  assert.equal(parseTimestamp('00:01:02.250'), 62.25);
  assert.equal(parseTimestamp('01:02.5'), 62.5);
  assert.equal(parseTimestamp('0:00:00.05'), 0.05);
  assert.equal(parseTimestamp('  1:00:00,000  '), 3600);
});

test('parseTimestamp 拒绝非法输入', () => {
  for (const bad of ['', 'abc', '61:00.000', '00:00:61,000', '1.5', '00:00:01']) {
    assert.equal(parseTimestamp(bad), null, `应拒绝: ${bad}`);
  }
});

test('格式化输出', () => {
  assert.equal(formatSrtTime(1.5), '00:00:01,500');
  assert.equal(formatSrtTime(3723.999), '01:02:03,999');
  assert.equal(formatVttTime(62.25), '00:01:02.250');
  assert.equal(formatAssTime(62.254), '0:01:02.25');
  assert.equal(formatAssTime(3661.2), '1:01:01.20');
  assert.equal(formatClock(62.25), '1:02.250');
  assert.equal(formatClock(3725.5), '1:02:05.500');
});

test('格式化对非有限时间早失败（RangeError）', () => {
  for (const fn of [formatSrtTime, formatVttTime, formatAssTime]) {
    for (const bad of [NaN, Infinity, -Infinity, 'x']) {
      assert.throws(() => fn(bad), RangeError, `${fn.name}(${String(bad)}) 应抛 RangeError`);
    }
  }
  // 负时间保持原有静默钳 0 行为
  assert.equal(formatSrtTime(-1), '00:00:00,000');
  assert.equal(formatAssTime(-0.5), '0:00:00.00');
});

test('parseTimestamp 毫秒超过 3 位截断', () => {
  assert.equal(parseTimestamp('00:00:01,1234'), 1.123);
  assert.equal(parseTimestamp('00:00:01.9999'), 1.999);
});

test('1000+ 小时 round-trip（小时位数不封顶）', () => {
  const t = parseTimestamp('1000:00:00,000');
  assert.equal(t, 3600000);
  assert.equal(formatSrtTime(t), '1000:00:00,000');
  assert.equal(formatVttTime(t), '1000:00:00.000');
  assert.equal(formatAssTime(t), '1000:00:00.00');
  assert.equal(parseTimestamp(formatSrtTime(t)), t);
});

test('parseFlexibleTime 面向行内编辑输入', () => {
  assert.equal(parseFlexibleTime('1:02.25'), 62.25);
  assert.equal(parseFlexibleTime('01:02.250'), 62.25);
  assert.equal(parseFlexibleTime('62.5'), 62.5);
  assert.equal(parseFlexibleTime('5'), 5);
  assert.equal(parseFlexibleTime('1:02:03.5'), 3723.5);
  assert.equal(parseFlexibleTime('bad'), null);
  assert.equal(parseFlexibleTime(''), null);
  assert.equal(parseFlexibleTime('1:75'), null);
});

test('parseFlexibleTime 接受无毫秒的 h:mm:ss（手打完整时分秒）', () => {
  assert.equal(parseFlexibleTime('0:00:03'), 3);
  assert.equal(parseFlexibleTime('0:00:03.500'), 3.5);
  assert.equal(parseFlexibleTime('1:02:03'), 3723);
  assert.equal(parseFlexibleTime('1000:00:00'), 3600000);
  // 越界字段仍拒绝
  assert.equal(parseFlexibleTime('0:60:00'), null);
  assert.equal(parseFlexibleTime('0:00:60'), null);
});
