import { useFloatingPanelStore, FloatingPanelLayer, FloatingPanelProvider } from '../src';

function DemoContent({ name }: { name: string }) {
  return (
    <div style={{ display: 'grid', placeItems: 'center', height: '100%', fontSize: 14 }}>
      这是「{name}」面板的内容，随便放任何 React 节点。
    </div>
  );
}

function Toolbar() {
  const store = useFloatingPanelStore();
  return (
    <div style={{ display: 'flex', gap: 8, padding: 24 }}>
      <button onClick={() => store.open({
        id: 'player',
        title: '播放器',
        width: 640,
        height: 400,
        content: <DemoContent name="播放器" />,
      })}>打开播放器</button>

      <button onClick={() => store.open({
        id: 'workorder',
        title: '工单中心',
        width: 760,
        height: 620,
        maximizable: true,
        content: <DemoContent name="工单中心" />,
      })}>打开工单（可最大化）</button>

      <button onClick={() => store.open({
        id: 'sticky',
        title: '便签',
        width: 280,
        height: 200,
        content: <DemoContent name="便签" />,
      })}>打开便签</button>

      <button onClick={() => store.closeAll()}>全部关闭</button>
    </div>
  );
}

/**
 * 用法：Provider 包住应用，Layer 放在 JSX 最后。
 * storageKey 让每个面板记住上次的位置和大小（localStorage）。
 */
export default function App() {
  return (
    <FloatingPanelProvider storageKey="demo-panels" workspacePadding={12}>
      <Toolbar />
      <FloatingPanelLayer />
    </FloatingPanelProvider>
  );
}
