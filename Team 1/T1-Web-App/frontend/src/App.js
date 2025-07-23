import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import * as d3 from 'd3';

//const BACKEND_URL = 'http://127.0.0.1:5000';
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  const [text, setText] = useState('');
  const [chunks, setChunks] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [results, setResults] = useState({
    chunkNumbers: [],
    cumulativeKeywords: [],
    kdr: [],
    pctChunk: [],
    entityDiscovery: [],
    convergenceChunk: null
  });

  const chunkScrollRef = useRef(); 

  useEffect(() => {
    const resetBackend = async () => {
      try {
        await fetch(`${BACKEND_URL}/reset`, { method: 'POST' });
        console.log("Backend state reset on load");
      } catch (err) {
        console.error("Backend reset failed on load:", err);
      }
    };
    resetBackend();
  }, []);

  const chunkTextByWords = (text, chunkSize = 600) => {
    const words = text.trim().split(/\s+/);
    const out = [];
    for (let i = 0; i < words.length; i += chunkSize) {
      out.push(words.slice(i, i + chunkSize).join(' '));
    }
    return out;
  };

  const handleSubmit = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/reset`, { method: 'POST' });
      const json = await res.json();
      if (json.status !== "reset") {
        console.warn("Unexpected reset response:", json);
        return;
      }
      console.log("Backend state reset");

      const generatedChunks = chunkTextByWords(text);
      setChunks(generatedChunks);
      setCurrentIndex(0);
      setResults({
        chunkNumbers: [],
        cumulativeKeywords: [],
        kdr: [],
        pctChunk: [],
        entityDiscovery: [],
        convergenceChunk: null
      });
    } catch (err) {
      console.error("Failed to reset backend:", err);
    }
  };

  const handleReset = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/reset`, { method: 'POST' });
      const json = await res.json();
      if (json.status === "reset") {
        console.log("Reset successful");
        setText('');
        setChunks([]);
        setCurrentIndex(0);
        setResults({
          chunkNumbers: [],
          cumulativeKeywords: [],
          kdr: [],
          pctChunk: [],
          entityDiscovery: [],
          convergenceChunk: null
        });
      }
    } catch (err) {
      console.error("Error resetting:", err);
    }
  };

  useEffect(() => {
    if (currentIndex < chunks.length) {
      const chunk = chunks[currentIndex];
      if (!chunk.trim()) return;

      const sendChunk = async () => {
        try {
          const response = await fetch(`${BACKEND_URL}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chunk })
          });

          const data = await response.json();
          console.log(`Chunk ${data.chunk} sent | Words: ${chunk.split(/\s+/).length}`);

          setResults((prev) => {
            const updated = {
              chunkNumbers: [...prev.chunkNumbers, data.chunk],
              cumulativeKeywords: [...prev.cumulativeKeywords, data.cumulative_keywords],
              kdr: [...prev.kdr, data.kdr],
              pctChunk: [...prev.pctChunk, data.random_value],
              entityDiscovery: [...prev.entityDiscovery, data.entity_relation_discovery],
              convergenceChunk: data.convergence_chunk ?? prev.convergenceChunk
            };

            //Auto-scroll to bottom of list
            setTimeout(() => {
              if (chunkScrollRef.current) {
                chunkScrollRef.current.scrollTop = chunkScrollRef.current.scrollHeight;
              }
            }, 100);

            return updated;
          });

          setCurrentIndex((prev) => prev + 1);
        } catch (err) {
          console.error("Chunk failed to send:", err);
        }
      };

      sendChunk();
    }
  }, [chunks, currentIndex]);

  return (
    <div className="app-container">
      {/* LEFT PANEL */}
      <div className="left-panel">
        <h1>T1-Speech Completion Estimation</h1>
        <textarea
          rows="10"
          cols="80"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste your text here"
        />
        <br />
        <button onClick={handleSubmit}>ANALYZE</button>
        <button onClick={handleReset} className="reset-button">RESET</button>

        {results.chunkNumbers.length > 0 && (
          <>
            <p>
              <strong>ESTIMATED CONVERGENCE AT CHUNK:</strong>{' '}
              {results.convergenceChunk ?? 'Not yet'}
            </p>
            <h3>CHUNK-WISE ESTIMATED PERCENTAGE COMPLETION</h3>

            <div className="chunk-scroll" ref={chunkScrollRef}>
              <ul>
                {results.pctChunk.map((val, i) => (
                  <li key={i}>
                    CHUNK {i + 1} ----- {val.toFixed(2)}
                  </li>
                ))}
              </ul>
            </div>
          </>
        )}
      </div>

      {/* RIGHT PANEL */}
      <div className="right-panel-container">
        <h2 className="right-panel-title">Analysis Charts</h2>
        <div className="right-panel">
          {results.chunkNumbers.length > 0 && (
            <div className="chart-grid">
              <div className="chart-box">
                <D3Chart
                  title="Cumulative Unique Keywords"
                  x={results.chunkNumbers}
                  y={results.cumulativeKeywords}
                  xLabel="Chunk"
                  yLabel="Cumulative Unique Keywords"
                />
              </div>

              <div className="chart-box">
                <D3Chart
                  title="Keyword Discovery Rate (KDR)"
                  x={results.chunkNumbers}
                  y={results.kdr}
                  xLabel="Chunk"
                  yLabel="KDR"
                  threshold={0.02}
                  convergence={results.convergenceChunk}
                />
              </div>

              <div className="chart-box">
                <D3Chart
                  title="Entity-Relation Discovery"
                  x={results.chunkNumbers}
                  y={results.entityDiscovery}
                  xLabel="Chunk"
                  yLabel="Entity-Relation Count"
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


function D3Chart({ title, x, y, xLabel, yLabel, threshold, convergence }) {
  const ref = useRef();

  useEffect(() => {
    const svgEl = ref.current;
    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();

    const container = svgEl.parentElement;
    const width = container.clientWidth;
    const height = container.clientHeight;
    const margin = { top: 30, right: 50, bottom: 50, left: 60 };

    const xScale = d3.scaleLinear()
      .domain([d3.min(x), d3.max(x)])
      .range([margin.left, width - margin.right]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(y) * 1.1 || 1])
      .range([height - margin.bottom, margin.top]);

    svg.append("g")
      .attr("transform", `translate(0, ${height - margin.bottom})`)
      .call(d3.axisBottom(xScale).ticks(Math.min(x.length, 8)).tickFormat(d3.format("d")));

    svg.append("g")
      .attr("transform", `translate(${margin.left}, 0)`)
      .call(d3.axisLeft(yScale).ticks(6));

    svg.append("path")
      .datum(y)
      .attr("fill", "none")
      .attr("stroke", "#0077cc")
      .attr("stroke-width", 2)
      .attr("d", d3.line()
        .x((_, i) => xScale(x[i]))
        .y((_, i) => yScale(y[i]))
        .curve(d3.curveMonotoneX)
      );

    svg.selectAll("circle")
      .data(y)
      .enter()
      .append("circle")
      .attr("cx", (_, i) => xScale(x[i]))
      .attr("cy", (_, i) => yScale(y[i]))
      .attr("r", 4)
      .attr("fill", "blue");

    if (threshold !== undefined) {
      svg.append("line")
        .attr("x1", margin.left)
        .attr("x2", width - margin.right)
        .attr("y1", yScale(threshold))
        .attr("y2", yScale(threshold))
        .attr("stroke", "red")
        .attr("stroke-dasharray", "5,5");
    }

    if (convergence !== undefined && convergence !== null) {
      svg.append("line")
        .attr("x1", xScale(convergence))
        .attr("x2", xScale(convergence))
        .attr("y1", margin.top)
        .attr("y2", height - margin.bottom)
        .attr("stroke", "green")
        .attr("stroke-dasharray", "5,5");
    }

    svg.append("text")
      .attr("x", width / 2)
      .attr("y", margin.top - 10)
      .attr("text-anchor", "middle")
      .attr("font-size", "14px")
      .attr("font-weight", "bold")
      .text(title);

    svg.append("text")
      .attr("x", width / 2)
      .attr("y", height - 10)
      .attr("text-anchor", "middle")
      .attr("font-size", "12px")
      .text(xLabel);

    svg.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -height / 2)
      .attr("y", 15)
      .attr("text-anchor", "middle")
      .attr("font-size", "12px")
      .text(yLabel);
  }, [x, y, xLabel, yLabel, title, threshold, convergence]);

  return (
  <svg
    ref={ref}
    viewBox="0 0 600 300"
    preserveAspectRatio="xMidYMid meet"
    style={{
      width: '100%',
      height: '100%',
      display: 'block'
    }}
  />
);
}



export default App;
