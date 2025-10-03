import { VercelV0Chat } from "./components/v0-ai-chat"

export default function Home() {
    return (
        <div className="min-h-screen flex items-center justify-center bg-white dark:bg-black">
            <VercelV0Chat />
        </div>
    )
}