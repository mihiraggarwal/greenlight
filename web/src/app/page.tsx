"use client";

import { Gauge, gaugeClasses } from '@mui/x-charts/Gauge';
import { BarChart } from '@mui/x-charts/BarChart';
import { useEffect, useState } from 'react';

export default function Home() {

  const sep_labels = ['Climate Change', 'Natural Capital', 'Pollution & Waste', 'Human Capital', 'Product Liability', 'Community Relations', 'Corporate Governance', 'Business Ethics & Values']
  const chatServer = "http://127.0.0.1:5000";

  const [company, setCompany] = useState('');
  const [rank, setRank] = useState(0);
  const [companies, setCompanies] = useState<string[]>([]);
  const [showModal, setShowModal] = useState(false);
  const [showDataModal, setShowDataModal] = useState(false);
  const [dataText, setDataText] = useState('');
  const [data, setData] = useState({
    greenwashing_score: {score: 0},
    flesch_reading_ease: {score: 0},
    sentiment_score: {score: 0},
    sentences_relative: {scores: {
      'environment': 0,
      'social': 0,
      'governance': 0,
    }},
    sentences_claims1: {sentences: []},
    sentences_action: {sentences: []},
    llm_action_contradicted: {texts: []},
    llm_not_action_contradicted: {texts: []},
    llm_separate_scores: {scores: {'Climate Change': 0, 'Natural Capital': 0, 'Pollution & Waste': 0, 'Human Capital': 0, 'Product Liability': 0, 'Community Relations': 0, 'Corporate Governance': 0, 'Business Ethics & Values': 0}}
  });

  const greenwashing_score = Math.round(data.greenwashing_score.score * 100);
  const flesch_reading_ease = Math.round(data.flesch_reading_ease.score * 100);
  const sentiment_score = Math.round(data.sentiment_score.score * 100);
  const environment = Math.round(data.sentences_relative.scores.environment * 100);
  const social = Math.round(data.sentences_relative.scores.social * 100);
  const governance = Math.round(data.sentences_relative.scores.governance * 100);

  useEffect(() => {
    fetch('/api/list')
      .then(res => res.json())
      .then(data => {
        setCompanies(data.data);
      })
      .catch(err => {
        console.error("Error fetching company list:", err);
      });
  }, []);

  return (
    <div className="flex flex-col items-center min-h-screen w-full pt-10 pb-20 pl-4 pr-4 gap-10 bg-gray-100">
      <div className='flex flex-col items-center w-full gap-3'>
        <h1 className="text-3xl font-bold text-black">Greenlight</h1>
        <p className="text-lg text-gray-500">Providing greenwashing scores and insights for companies</p>
      </div>
      <select className="w-full sm:w-2/5 p-2 border border-gray-300 rounded text-black" onChange={async (e) => {
          setCompany(e.target.value.toLowerCase())
          const res = await fetch('/api/company',  {
            method: 'POST',
            body: JSON.stringify({company: e.target.value.toLowerCase()}),
          })
          const data = await res.json();
          setData(data.data);
          setRank(data.rank);
        }}>
        <option value="">Select a company...</option>
        {companies.map((company, index) => (
          <option key={index} value={company}>{company.length <= 3 ? company.toUpperCase() : `${company.charAt(0).toUpperCase()}${company.slice(1)}`}</option>
        ))}
      </select>
      { company && (
        <div className="flex flex-col items-center w-full sm:w-2/5 gap-8">
          <div className="flex flex-row justify-between items-center w-full">
            <Gauge width={150} height={150} value={greenwashing_score} sx={() => ({
              [`& .${gaugeClasses.valueText}`]: {
                fontSize: 20,
              },
              [`& .${gaugeClasses.valueArc}`]: {
                fill: greenwashing_score < 25 ? '#52b202' : '#ff0000',
              },
            })}
            />
            <div className="text-lg text-gray-700">
              <p><span className='font-bold'>Company</span>: {company.toUpperCase()}</p>
              <p><span className='font-bold'>Greenwashing Score</span>: {greenwashing_score}</p>
              <p><span className='font-bold'>Rank</span>: {rank}</p>
            </div>
          </div>
          <h2 className="text-2xl font-bold text-black text-left w-full">Insights</h2>
          <div className="flex flex-col items-center w-full gap-8">
            <div className="flex flex-row justify-between items-center w-full bg-white shadow-md rounded-lg p-4 mb-4">
              <div className="flex flex-col items-center justify-center">
                <Gauge width={100} height={100} value={flesch_reading_ease} />
                <p className="text-lg text-gray-700">Readibility Score</p>
              </div>
              <div className="flex flex-col items-center justify-center">
                <Gauge width={100} height={100} value={sentiment_score} />
                <p className="text-lg text-gray-700">Sentiment Score</p>
              </div>
            </div>
            <div className='flex flex-col gap-1 w-full items-center'>
              <div className='w-full mr-5'>
                <BarChart
                  yAxis={[{ scaleType: 'band', data: ['Env', 'Soc', 'Gov'] }]}
                  series={[{ data: [environment, social, governance] }]}
                  height={300}
                  // width={550}
                  margin={{ top: 0, bottom: 0, left: 0, right: 0 }}
                  layout='horizontal'
                />
              </div>
              <p className="text-lg text-gray-700">Relative ESG Scores</p>
            </div>
            <div className="flex flex-col sm:flex-row w-full sm:w-auto justify-between items-center p-4 gap-6">
              <div className="flex flex-col justify-between items-center bg-white shadow-md rounded-lg p-4 w-full gap-2">
                <p className="text-2xl font-bold text-gray-700">{data.sentences_claims1.sentences.length}</p>
                <p className="text-lg text-gray-500">Sustainability Claims</p>
              </div>
              <div className="flex flex-col justify-between items-center bg-white shadow-md rounded-lg p-4 w-full gap-2">
                <p className="text-2xl font-bold text-gray-700">{data.sentences_action.sentences.length}</p>
                <p className="text-lg text-gray-500">Sustainability Actions</p>
              </div>
              <div className="flex flex-col justify-between items-center bg-white shadow-md rounded-lg p-4 w-full gap-2">
                <p className="text-2xl font-bold text-gray-700">{data.llm_action_contradicted.texts.length + data.llm_not_action_contradicted.texts.length}</p>
                <p className="text-lg text-gray-500">News Contradictions</p>
              </div>
            </div>
            {/* sep and sec */}
            <div className="w-full mr-5">
              <BarChart
                xAxis={[{ scaleType: 'band', data: sep_labels }]}
                series={[{ data: Object.values(data.llm_separate_scores.scores) }]}
                height={300}
                // width={550}
                margin={{ top: 0, bottom: 0, left: 0, right: 0 }}
              />
            </div>
            <div className='flex flex-col w-full gap-2 items-center'>
              <div className="flex flex-row justify-between items-center w-full gap-2">
                <button className="bg-blue-500 text-white px-4 py-2 rounded w-full cursor-pointer" onClick={async () => {
                  setShowDataModal(true);
                  const res = await fetch(`${chatServer}/master`, {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ company: company })
                  })
                  const json = await res.json();
                  setDataText(JSON.stringify(json));
                }}>Master Data</button>
                <button className="bg-blue-500 text-white px-4 py-2 rounded w-full cursor-pointer" onClick={async () => {
                  setShowDataModal(true);
                  const res = await fetch(`${chatServer}/logfile`, {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ company: company })
                  })
                  const json = await res.json();
                  setDataText(JSON.stringify(json));
                }}>Logfile</button>
              </div>
              <button className="bg-blue-500 text-white px-4 py-2 rounded w-full cursor-pointer" onClick={() => setShowModal(true)}>Ask Questions about the Company</button>
            </div>
            <p className='text-gray-500 text-sm'><span className='font-bold'>Disclaimer</span>: The greenwashing detection results are AI-generated and should be used as a tool for analysis, not as definitive judgments. Results may contain inaccuracies and should be interpreted with caution.</p>
          </div>
        </div>
      )}
      {/* footer */}
      <footer className="w-full text-center fixed bottom-0 p-3 text-sm bg-gray-100 text-black border-t-1">
        <p>Created with &lt;3 by <a href="https://mihiraggarwal.me" target='_blank' className='underline text-blue-400'>Mihir Aggarwal</a></p>
      </footer>
      {/* modals */}
      {showModal && <Modal company={company.toLowerCase()} setShowModal={setShowModal} />}
      {showDataModal && <DataModal text={dataText} setShowDataModal={setShowDataModal} />}
    </div>
  )
}

const Modal = ({company, setShowModal}: {company: string, setShowModal: (value: boolean) => void} ) => {

  const [input, setInput] = useState('');
  const [chat, setChat] = useState<{ type: string; text: string }[]>([]);
  const [loading, setLoading] = useState(false);

  const chatServer = "http://127.0.0.1:5000";

  const Chat = () => {
    return <>
      {chat.map((item, index) => {
        if (item.type === 'question') {
          return (
            <div key={index} className="text-black w-full flex flex-row justify-between">
              <div></div>
              <div className='border-2 p-2 rounded-2xl w-auto bg-gray-200 max-w-3/4'>{item.text}</div>
            </div>
          );
        } else {
          return (
            <div key={index} className="text-black w-full flex flex-row justify-between">
              <div className='border-2 p-2 rounded-2xl w-auto bg-blue-200 max-w-3/4'>{item.text}</div>
              <div></div>
            </div>
          );
        }
      })}
    </>
  }

  const Loading = () => {
    return (
      <div className="text-black w-full flex flex-row justify-between">
        <div className='border-2 p-2 rounded-2xl w-auto max-w-3/4'>Loading...</div>
        <div></div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full flex items-center justify-center">
    <div className="p-8 border w-2/3 h-2/3 shadow-lg rounded-md bg-white flex flex-col items-center justify-between">
      <div className="text-center">
        <h3 className="text-2xl font-bold text-gray-900">Ask Questions about the Company</h3>
      </div>
      <div className='flex flex-col justify-start items-start w-full h-full overflow-y-scroll mt-4 p-4 gap-2'>
        <Chat />
        {loading && <Loading />}
      </div>
      <div className='flex flex-row justify-between w-full mt-4'>
        <input value={input} type="text" className="border-2 p-2 rounded w-full text-black" placeholder="Ask a question..." onChange={(e) => setInput(e.target.value)} />
        <button className="ml-4 bg-blue-500 text-white px-4 py-2 rounded" onClick={async (e) => {
          console.log(e.target)
          setChat([...chat, {
            type: 'question',
            text: input
          }])
          setInput('')
          setLoading(true);
          const res = await fetch(`${chatServer}/`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: input, company: company })
          })
          const data = await res.json();
          setLoading(false);
          setChat([...chat, {
            type: 'question',
            text: input
          },{
            type: 'answer',
            text: data.answer
          }])
        }}>
          Ask
        </button>
      </div>
      <button className="mt-4 bg-blue-500 text-white px-4 py-2 rounded" onClick={() => setShowModal(false)}>Close</button>
    </div>
  </div>
  )
}

const DataModal = ({text, setShowDataModal}: {text: string, setShowDataModal: (value: boolean) => void}) => {
  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full flex items-center justify-center text-black">
    <div className="p-8 border w-2/3 h-2/3 shadow-lg rounded-md bg-white flex flex-col items-center justify-between">
      <div className="text-center">
        <h3 className="text-2xl font-bold text-gray-900">master.json</h3>
      </div>
      <pre className='flex flex-col justify-start items-start w-full h-full overflow-y-scroll mt-4 p-4 gap-2 whitespace-pre-line'>
        {text}
      </pre>
      <button className="mt-4 bg-blue-500 text-white px-4 py-2 rounded" onClick={() => setShowDataModal(false)}>Close</button>
    </div>
  </div>
  )
}