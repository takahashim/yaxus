// The translation, laid out as it arrives. Segments may be translated side by side, so the
// text of each is kept apart and the output is made from all of them in order: a segment
// owes a separator (`breaks[i]`: a blank line after a paragraph, a space or nothing between
// sentences) before its text, which is dropped for the first text, and a blank line is kept
// even when the segment before it has none.
export class Layout {
  constructor(el) {
    this.el = el;
    this.breaks = [];
    this.parts = [];
    this.frame = 0;
  }

  plan(breaks) {
    this.breaks = breaks;
    this.parts = breaks.map(() => '');
  }

  add(segment, text) {
    this.parts[segment] += text;
    // One redraw per frame, however many pieces arrived.
    if (!this.frame) this.frame = requestAnimationFrame(() => this.flush());
  }

  flush() {
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = 0;
    let out = '';
    let owed = '';
    this.parts.forEach((part, i) => {
      if (out && (owed === '' || this.breaks[i] === '\n\n')) owed = this.breaks[i];
      if (part) {
        out += owed + part;
        owed = '';
      }
    });
    this.el.textContent = out;
    this.el.scrollTop = this.el.scrollHeight;
  }
}
