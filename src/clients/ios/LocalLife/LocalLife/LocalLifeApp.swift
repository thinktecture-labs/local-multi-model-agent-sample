//
//  LocalLifeApp.swift
//  LocalLife
//
//  Created by Christian Weyer on 03.03.26.
//

import SwiftUI

@main
struct LocalLifeApp: App {
    @State private var viewModel = ChatViewModel()

    var body: some Scene {
        WindowGroup {
            ChatView()
                .environment(viewModel)
                .task { await viewModel.setup() }
        }
    }
}
