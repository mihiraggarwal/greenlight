'use client'

import { useState } from 'react'
import { toast } from 'react-hot-toast'

export default function Add() {
  const [companyName, setCompanyName] = useState('')
	const [url1, setUrl1] = useState('')
	const [url2, setUrl2] = useState('')
  const [annualReport, setAnnualReport] = useState<File | null>(null)
  const [esgReport, setEsgReport] = useState<File | null>(null)

	const MAIN_SERVER = "http://localhost:5000";

  const handleSubmit = async () => {
		if (!companyName || !annualReport || !url1) {
			toast.error('Please fill all required fields.')
			return
		}

		const formData = new FormData()
		formData.append('company', companyName.toLowerCase())
		formData.append('annual', annualReport)
		formData.append('url1', url1)
		formData.append('url2', url2 || '')

		if (esgReport) {
			formData.append('esg', esgReport)
		}

		toast.success('Uploading and initiating analysis...')

		try {
			const res = await fetch(`${MAIN_SERVER}/submit`, {
				method: 'POST',
				body: formData
			})

			if (res.ok) {
				toast('Your data has been submitted! The results should be visible in a few hours.')
			} else {
				toast.error('Failed to upload. Please check the server logs.')
			}

		} catch (err) {
				toast.error('Failed to upload. Please try again later.')
				console.error('Upload error:', err)
		}
	}

  return (
		<div className="min-h-screen w-full bg-gray-50 text-gray-800 px-6 py-10 flex flex-col items-center gap-10">
			<div className='flex flex-col items-center w-full gap-3'>
				<h1 className="text-3xl font-bold text-black">Greenlight</h1>
				<p className="text-lg text-gray-500">Providing greenwashing scores and insights for companies</p>
			</div>

			<div className="w-full max-w-xl bg-white p-6 rounded-lg shadow-md flex flex-col gap-4">
				<label className="block mb-2 text-lg font-semibold text-gray-700">Add a Company</label>
				<input
					type="text"
					placeholder="Company Name*"
					value={companyName}
					onChange={(e) => setCompanyName(e.target.value)}
					className="border p-2 rounded-md w-full"
					required
				/>

				<input
					type="text"
					placeholder="Company Website URL*"
					value={url1}
					onChange={(e) => setUrl1(e.target.value)}
					className="border p-2 rounded-md w-full"
					required
				/>

				<input
					type="text"
					placeholder="Company Website URL (Alternate)"
					value={url2}
					onChange={(e) => setUrl2(e.target.value)}
					className="border p-2 rounded-md w-full"
					required
				/>

				<div>
					<label className="block mb-1 text-sm font-medium text-gray-700">Annual Report (PDF) <span className='text-red-500'>*</span></label>
					<input
						type="file"
						accept="application/pdf"
						onChange={(e) => setAnnualReport(e.target.files?.[0] || null)}
						className="w-full file:bg-gray-300 file:px-3 file:py-1 file:rounded-sm file:text-black file:hover:cursor-pointer file:font-medium file:text-sm file:border-0 file:mr-2 file:mb-2"
					/>
				</div>

				<div>
					<label className="block mb-1 text-sm font-medium text-gray-700">ESG Report (PDF)</label>
					<input
						type="file"
						accept="application/pdf"
						onChange={(e) => setEsgReport(e.target.files?.[0] || null)}
						className="w-full file:bg-gray-300 file:px-3 file:py-1 file:rounded-sm file:text-black file:hover:cursor-pointer file:font-medium file:text-sm file:border-0 file:mr-2 file:mb-2"
					/>
				</div>

				<button
					className="mt-4 bg-blue-600 hover:bg-blue-700 hover:cursor-pointer text-white px-4 py-2 rounded-md"
					onClick={handleSubmit}
				>
					Submit for Analysis
				</button>
			</div>
		</div>
  )
}
