import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';

interface Props {
  label: string;
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/** 单个面板内容崩溃时不拖垮整个工作区。 */
export class PanelErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(`[floating-panel-kit] 面板「${this.props.label}」渲染失败:`, error, info.componentStack);
  }

  render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="fp-error" role="alert">
          面板「{this.props.label}」渲染失败：{this.state.error.message}
        </div>
      );
    }
    return this.props.children;
  }
}
