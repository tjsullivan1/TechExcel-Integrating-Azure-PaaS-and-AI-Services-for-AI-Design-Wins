﻿using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;

namespace ContosoSuitesWebAPI.Agents
{
    // Exercise 5 Task 2 TODO #2: Inject the Kernel service into the MaintenanceCopilot class.
    /// <summary>
    /// The maintenance copilot agent for assisting with maintenance requests.
    /// </summary>
    public class MaintenanceCopilot(Kernel kernel)
    {
        // Exercise 5 Task 2 TODO #3: Uncomment the two lines below to declare the Kernel and ChatHistory objects.
        public readonly Kernel _kernel = kernel;
        private ChatHistory _history = new("""
            You are a helpful maintenance copilot. You intake requests from luxury hotel guests for the hotel maintenance team.
            You should ensure you have all the necessary information to assist with maintenance requests.
            You should ensure you have permission to perform additional actions, such as saving the request to the database. Please ask the user to confirm before saving.
            You should inform the user that you have saved the requet and maintenance will address them shortly.
            """);

        /// <summary>
        /// Chat with the maintenance copilot.
        /// </summary>
        public async Task<string> Chat(string userPrompt)
        {
            // Exercise 5 Task 2 TODO #4: Comment out or delete the throw exception line below,
            // and then uncomment the remaining code in the function.
            //throw new NotImplementedException();

            var chatCompletionService = _kernel.GetRequiredService<IChatCompletionService>();

            var openAIPromptExecutionSettings = new OpenAIPromptExecutionSettings()
            {
                ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions
            };

            _history.AddUserMessage(userPrompt);

            var result = await chatCompletionService.GetChatMessageContentAsync(
                _history,
                executionSettings: openAIPromptExecutionSettings,
                _kernel
            );

            _history.AddAssistantMessage(result.Content!);

            return result.Content!;
        }
    }
}
