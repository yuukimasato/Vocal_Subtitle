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
